#include "colza.H"

#include "amr-wind/CFDSim.H"
#include "amr-wind/utilities/io_utils.H"

#include "AMReX_ParmParse.H"
#include "AMReX_Conduit_Blueprint.H"

#include <ascent.hpp>
#include <colza/Client.hpp>
#include <colza/MPIClientCommunicator.hpp>
#include <ssg.h>

namespace amr_wind {
namespace colza_int {

ColzaPostProcess::ColzaPostProcess(CFDSim& sim, const std::string& label)
    : m_sim(sim), m_label(label), m_colza_comm(amrex::ParallelDescriptor::Communicator())
{}

ColzaPostProcess::~ColzaPostProcess()
{
    BL_PROFILE("amr-wind::ColzaPostProcess::~ColzaPostProcess");

    m_colza_pipeline = colza::DistributedPipelineHandle();
    m_colza_client = colza::Client();
    ssg_finalize();
    m_thallium_engine.finalize();
}

void ColzaPostProcess::pre_init_actions() {}

void ColzaPostProcess::initialize()
{
    BL_PROFILE("amr-wind::ColzaPostProcess::initialize");

    amrex::Vector<std::string> field_names;

    std::string colza_protocol;
    int         colza_provider_id{0};
    std::string colza_ssg_file;
    std::string colza_pipeline_name;

    {
        amrex::ParmParse pp("colza");
        pp.getarr("fields", field_names);
        pp.query("output_frequency", m_out_freq);
        pp.get("protocol", colza_protocol);
        pp.query("provider_id", colza_provider_id);
        pp.get("ssg_file", colza_ssg_file);
        pp.get("pipeline_name", colza_pipeline_name);
    }

    // Process field information
    auto& repo = m_sim.repo();

    for (const auto& fname : field_names) {
        if (!repo.field_exists(fname)) {
            amrex::Print() << "WARNING: Colza: Non-existent field requested: "
                           << fname << std::endl;
            continue;
        }

        auto& fld = repo.get_field(fname);
        m_fields.emplace_back(&fld);
        ioutils::add_var_names(m_var_names, fld.name(), fld.num_comp());
    }

    // Initialize thallium, Colza client, and pipeline handle
    m_thallium_engine = thallium::engine(colza_protocol, THALLIUM_SERVER_MODE, false, 0);
    ssg_init();
    m_colza_client = colza::Client(m_thallium_engine);
    m_colza_pipeline = m_colza_client.makeDistributedPipelineHandle(
        &m_colza_comm, colza_ssg_file, (uint16_t)colza_provider_id, colza_pipeline_name);
}

void ColzaPostProcess::post_advance_work()
{
    BL_PROFILE("amr-wind::ColzaPostProcess::post_advance_work");

    const auto& time = m_sim.time();
    const int tidx = time.time_index();
    // Output only on given frequency
    if (!(tidx % m_out_freq == 0)) return;

    amrex::Vector<int> istep(
        m_sim.mesh().finestLevel() + 1, m_sim.time().time_index());

    int plt_num_comp = 0;
    for (auto* fld : m_fields) {
        plt_num_comp += fld->num_comp();
    }

    auto outfield = m_sim.repo().create_scratch_field(plt_num_comp);

    const int nlevels = m_sim.repo().num_active_levels();

    for (int lev = 0; lev < nlevels; ++lev) {
        int icomp = 0;
        auto& mf = (*outfield)(lev);

        for (auto* fld : m_fields) {
            amrex::MultiFab::Copy(
                mf, (*fld)(lev), 0, icomp, fld->num_comp(), 0);
            icomp += fld->num_comp();
        }
    }

    const auto& mesh = m_sim.mesh();

    amrex::Print() << "Calling Colza at time " << m_sim.time().new_time()
                   << std::endl;
    conduit::Node bp_mesh;
    amrex::MultiLevelToBlueprint(
        nlevels, outfield->vec_const_ptrs(), m_var_names, mesh.Geom(),
        m_sim.time().new_time(), istep, mesh.refRatio(), bp_mesh);

    conduit::Node verify_info;
    if (!conduit::blueprint::mesh::verify(bp_mesh, verify_info)) {
        ASCENT_INFO("Error: Mesh Blueprint Verify Failed!");
        verify_info.print();
    }

    int rank;
    MPI_Comm_rank(amrex::ParallelDescriptor::Communicator(), &rank);

    auto mesh_str = bp_mesh.to_string("conduit_base64_json", 0, 0, "", "");

    std::vector<size_t>  dimensions = { mesh_str.size()+1 };
    std::vector<int64_t> offsets    = { 0 };

    int32_t result;
    m_colza_pipeline.start((uint64_t)tidx);

    m_colza_pipeline.stage("mesh", (uint64_t)tidx, rank, dimensions, offsets,
                           colza::Type::UINT8, mesh_str.c_str(),
                           &result);

    m_colza_pipeline.execute((uint64_t)tidx);

    m_colza_pipeline.cleanup((uint64_t)tidx);
}

void ColzaPostProcess::post_regrid_actions()
{
    // nothing to do here
}

} // namespace ascent_int
} // namespace amr_wind

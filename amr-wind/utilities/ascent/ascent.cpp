#include "ascent.H"

#include "amr-wind/CFDSim.H"
#include "amr-wind/utilities/io_utils.H"

#include "AMReX_ParmParse.H"
#include "AMReX_Conduit_Blueprint.H"

#include <ascent.hpp>
#include <tclap/CmdLine.h>
#include <ams/Client.hpp>


namespace tl = thallium;
namespace amr_wind {
namespace ascent_int {

static std::string g_address_file;
static std::string g_address;
static std::string g_protocol;
static std::string g_node;
static unsigned    g_provider_id;
static std::string g_log_level = "info";
int use_local = 1;

static void parse_command_line();

/* Helper function to read and parse input args */
static std::string read_nth_line(const std::string& filename, int n)
{
   std::ifstream in(filename.c_str());

   std::string s;
   //for performance
   s.reserve(200);

   //skip N lines
   for(int i = 0; i < n; ++i)
       std::getline(in, s);

   std::getline(in,s);
   return s;
}

void parse_command_line() {
	int rank = 0;
        int size = 1;
#ifdef BL_USE_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif

	char *addr_file_name = getenv("AMS_SERVER_ADDR_FILE");
	char *node_file_name = getenv("AMS_NODE_ADDR_FILE");
	char *use_local_opt = getenv("AMS_USE_LOCAL_ASCENT");

	/* The logic below grabs the server address corresponding the client's MPI rank (MXM case) */
	size_t pos = 0;
        g_address_file = std::string(addr_file_name);
	std::string delimiter = " ";
	std::string l = read_nth_line(g_address_file, rank+1);
	pos = l.find(delimiter);
	std::string server_rank_str = l.substr(0, pos);
	std::stringstream s_(server_rank_str);
	int server_rank;
	s_ >> server_rank;
	assert(server_rank == rank);
	l.erase(0, pos + delimiter.length());
	g_address = l;
	std::string use_local_str = use_local_opt;
	std::stringstream s__(use_local_str);
	s__ >> use_local;

        g_provider_id = 0;
        g_node = read_nth_line(std::string(node_file_name), rank);
        g_protocol = g_address.substr(0, g_address.find(":"));
}

AscentPostProcess::AscentPostProcess(CFDSim& sim, const std::string& label)
    : m_sim(sim), m_label(label)
{}

AscentPostProcess::~AscentPostProcess() = default;

void AscentPostProcess::pre_init_actions() {}

void AscentPostProcess::initialize()
{
    BL_PROFILE("amr-wind::AscentPostProcess::initialize");

    amrex::Vector<std::string> field_names;

    {
        amrex::ParmParse pp("ascent");
        pp.getarr("fields", field_names);
        pp.query("output_frequency", m_out_freq);
    }

    // Process field information
    auto& repo = m_sim.repo();

    for (const auto& fname : field_names) {
        if (!repo.field_exists(fname)) {
            amrex::Print() << "WARNING: Ascent: Non-existent field requested: "
                           << fname << std::endl;
            continue;
        }

        auto& fld = repo.get_field(fname);
        m_fields.emplace_back(&fld);
        ioutils::add_var_names(m_var_names, fld.name(), fld.num_comp());
    }
}

void AscentPostProcess::post_advance_work()
{
    static int ams_initialized = 0;
    if(!ams_initialized) {
        /*Connect to server */
        parse_command_line();
	ams_initialized = 1;
    }
    BL_PROFILE("amr-wind::AscentPostProcess::post_advance_work");

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

    amrex::Print() << "Calling Ascent at time " << m_sim.time().new_time()
                   << std::endl;
    conduit::Node bp_mesh;
    amrex::MultiLevelToBlueprint(
        nlevels, outfield->vec_const_ptrs(), m_var_names, mesh.Geom(),
        m_sim.time().new_time(), istep, mesh.refRatio(), bp_mesh);

    ascent::Ascent ascent;
    conduit::Node open_opts;
    // Initialize a Client
    tl::engine engine(g_protocol, THALLIUM_CLIENT_MODE);
    ams::Client client(engine);

    // Open the Database "mydatabase" from provider 0
    ams::NodeHandle node =
        client.makeNodeHandle(g_address, g_provider_id,
                ams::UUID::from_string(g_node.c_str()));

    node.sayHello();


#ifdef BL_USE_MPI
    std::cout << "Do I get invoked???" << std::endl;
    open_opts["mpi_comm"] =
        MPI_Comm_c2f(amrex::ParallelDescriptor::Communicator());
#endif

    if(!use_local) {
	std::cout << "Using Ascent microservice!" << std::endl;
        node.ams_open(open_opts);
    } else {
        ascent.open();
	std::cout << "Using local Ascent!" << std::endl;
    }

    conduit::Node verify_info;
    if (!conduit::blueprint::mesh::verify(bp_mesh, verify_info)) {
        ASCENT_INFO("Error: Mesh Blueprint Verify Failed!");
        verify_info.print();
    }

    conduit::Node actions;


    /* This is an RPC call. What happens under the hood is: 
     * 1. Convert bp_mesh (a conduit "Node") to a string representation using bp_mesh.to_string()
     * 2. Check the size of the string thus created --- too large? RDMA. Else: inline RPC argument
     * 3. Send RPC call. This can be one-sided (asynchronous) ! Very fast as I do not need a response from server.*/ 
    if(!use_local) {
        node.ams_publish_and_execute(bp_mesh, actions);
    } else {
	ascent.publish(bp_mesh);
	ascent.execute(actions);
    }


    if(!use_local) {
        node.ams_close();
    } else {
	ascent.close();
    }
}

void AscentPostProcess::post_regrid_actions()
{
    // nothing to do here
}

} // namespace ascent_int
} // namespace amr_wind

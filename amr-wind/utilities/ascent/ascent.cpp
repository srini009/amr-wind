#include "ascent.H"

#include "amr-wind/CFDSim.H"
#include "amr-wind/utilities/io_utils.H"

#include "AMReX_ParmParse.H"
#include "AMReX_Conduit_Blueprint.H"

#include <ascent.hpp>
#include <unistd.h>
#include <tclap/CmdLine.h>
#include <ams/Client.hpp>
#include <conduit_blueprint.hpp>
#include <conduit_blueprint_mesh_utils.hpp>
#include <conduit_blueprint_mpi_mesh.hpp>
#include <abt.h>
#include <sys/time.h>
#include <unistd.h>

namespace tl = thallium;
namespace amr_wind {
namespace ascent_int {

static std::string g_address_file;
static std::string g_address;
static std::string g_protocol = "na+sm";
static std::string g_node;
static unsigned    g_provider_id;
static std::string g_log_level = "info";
int use_local = 1;
int i_should_participate_in_server_calls = 0;
int num_server = 1;
tl::engine *engine;
ams::Client *client;
ams::NodeHandle ams_client;
std::vector<tl::async_response> areq_array;
int current_buffer_index = 0;

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
	char *num_servers = getenv("AMS_NUM_SERVERS");

	/* The logic below grabs the server address corresponding the client's MPI rank (MXM case) */
	std::string num_servers_str = num_servers;
	std::stringstream n_(num_servers_str);
	n_ >> num_server;
	std::string use_local_str = use_local_opt;
	std::stringstream s__(use_local_str);
	s__ >> use_local;

	if(rank + 1 <= num_server) {
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
	        g_provider_id = 0;
	        g_node = read_nth_line(std::string(node_file_name), rank);
	        g_protocol = g_address.substr(0, g_address.find(":"));
		i_should_participate_in_server_calls = 1;
	}
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

static void wait_for_pending_requests()
{
    for(auto i = areq_array.begin(); i != areq_array.end(); i++) {
        bool ret;
        i->wait();
    }
}

void AscentPostProcess::post_advance_work()
{
    static int ams_initialized = 0;
    if(!ams_initialized) {
        /*Connect to server */
	sleep(5);
        parse_command_line();
	ams_initialized = 1;
	engine = new tl::engine(g_protocol, THALLIUM_CLIENT_MODE);
	client = new ams::Client(*engine);
    	if(i_should_participate_in_server_calls) {
    	    // Initialize a Client
            ams_client = (*client).makeNodeHandle(g_address, g_provider_id,
            	ams::UUID::from_string(g_node.c_str()));
    	}
    }
    double start = MPI_Wtime();
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

    //Add current directory to open opts
    open_opts["default_dir"] = getenv("AMS_WORKING_DIR");
    open_opts["actions_file"] = getenv("AMS_ACTIONS_FILE");
    open_opts["task_id"] = std::stoi(std::string(getenv("AMS_TASK_ID")));

    int my_rank = 0;
#ifdef BL_USE_MPI
    MPI_Comm_rank(amrex::ParallelDescriptor::Communicator(), &my_rank);
    open_opts["mpi_comm"] =
        MPI_Comm_c2f(amrex::ParallelDescriptor::Communicator());
#endif

    conduit::Node verify_info;
    if (!conduit::blueprint::mesh::verify(bp_mesh, verify_info)) {
        ASCENT_INFO("Error: Mesh Blueprint Verify Failed!");
        verify_info.print();
    }

    /* Mesh partitioning using Conduit */
    conduit::Node partitioned_mesh;
    conduit::Node partitioning_options;
    /*if(!use_local) {
        partitioning_options["target"] = num_server;
	conduit::blueprint::mpi::mesh::partition(bp_mesh, partitioning_options, partitioned_mesh, amrex::ParallelDescriptor::Communicator());
    }*/

    conduit::Node actions;

    /* This is an RPC call. What happens under the hood is: 
     * 1. Convert open_opts, bp_mesh, and actions (a conduit "Node") to a string representation using conduit::Node.to_string()
     * 2. Send RPC call. This can be made one-sided (asynchronous) if needed*/
    MPI_Barrier(amrex::ParallelDescriptor::Communicator());

    unsigned int ts, min_ts;
    //high resolution timestamp in microsecond
    struct timeval tv;
    gettimeofday(&tv, NULL);
    ts = (unsigned int)(tv.tv_sec * 1000 + tv.tv_usec / 1000);

    /* Min value of ts */
    MPI_Allreduce(&ts, &min_ts, 1, MPI_UNSIGNED, MPI_MIN, amrex::ParallelDescriptor::Communicator());

    if(!use_local and i_should_participate_in_server_calls) {
        auto response = ams_client.ams_open_publish_execute(open_opts, bp_mesh, bp_mesh.to_string("conduit_json").size(), actions, min_ts);
	areq_array.push_back(std::move(response));
    } else if(use_local) {

        ascent.open(open_opts);
	ascent.publish(bp_mesh);
	ascent.execute(actions);
	ascent.close();
    }

    MPI_Barrier(amrex::ParallelDescriptor::Communicator());

    double end = MPI_Wtime();
    if(my_rank == 0)
        std::cout << "Total time for ascent call on process: " << end-start << std::endl;

    current_buffer_index += 1;

    /* Before I exit, checking for pending requests sitting around */
    if(current_buffer_index == 101) {
       wait_for_pending_requests();
       MPI_Barrier(amrex::ParallelDescriptor::Communicator());
       if(my_rank == 0) {
           std::cerr << "Task ID: " << std::stoi(std::string(getenv("AMS_TASK_ID"))) << " is done." << std::endl;
       }
    }

}

void AscentPostProcess::post_regrid_actions()
{
    // nothing to do here
}

} // namespace ascent_int
} // namespace amr_wind

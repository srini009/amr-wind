#ifndef COLZA_INT_H
#define COLZA_INT_H

#include "amr-wind/utilities/PostProcessing.H"
#include <colza/Client.hpp>
#include <colza/MPIClientCommunicator.hpp>

/**
 * Colza In-situ Integration with Ascent pipelines
 */

namespace amr_wind {

class Field;

namespace colza_int {

class ColzaPostProcess : public PostProcessBase::Register<ColzaPostProcess>
{
public:
    static const std::string identifier() { return "Colza"; }

    ColzaPostProcess(CFDSim&, const std::string&);

    virtual ~ColzaPostProcess();

    //! Perform actions before mesh is created
    void pre_init_actions() override;

    //! Read user inputs and create the different data probe instances
    void initialize() override;

    //! Interpolate fields at a given timestep and output to disk
    void post_advance_work() override;

    //! Actions to perform post regrid e.g. redistribute particles
    void post_regrid_actions() override;

protected:
private:
    CFDSim& m_sim;
    std::string m_label;

    amrex::Vector<std::string> m_var_names;
    amrex::Vector<Field*> m_fields;

    int m_out_freq{1};

    thallium::engine                 m_thallium_engine;
    colza::MPIClientCommunicator     m_colza_comm;
    colza::Client                    m_colza_client;
    colza::DistributedPipelineHandle m_colza_pipeline;
};

} // namespace colza_int
} // namespace amr_wind

#endif

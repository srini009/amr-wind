#ifndef SMAGORINSKY_H
#define SMAGORINSKY_H

#include <string>
#include "amr-wind/turbulence/TurbModelBase.H"
#include "amr-wind/core/FieldRepo.H"

namespace amr_wind {
namespace turbulence {

/** Smagorinsky LES Model
 *  \ingroup turb_model
 */
template <typename Transport>
class Smagorinsky : public TurbModelBase<Transport>
{
public:
    static std::string identifier()
    {
        return "Smagorinsky-" + Transport::identifier();
    }

    explicit Smagorinsky(CFDSim& sim);

    //! Model name for debugging purposes
    virtual std::string model_name() const override { return "Smagorinsky"; }

    //! Update the turbulent viscosity field
    virtual void update_turbulent_viscosity(const FieldState fstate) override;

    //! No post advance work for this model
    virtual void post_advance_work() override {}

    //! Return model coefficients dictionary
    TurbulenceModel::CoeffsDictType model_coeffs() const override;

private:
    //! Smagorinsky coefficient (default value set for ABL simulations)
    amrex::Real m_Cs{0.135};

    const Field& m_vel;
    const Field& m_rho;
};

} // namespace turbulence
} // namespace amr_wind

#endif /* SMAGORINSKY_H */

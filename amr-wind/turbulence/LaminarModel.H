#ifndef LAMINARMODEL_H
#define LAMINARMODEL_H

#include "amr-wind/turbulence/TurbModel.H"

namespace amr_wind {
namespace turbulence {

/** Laminar model
 *  \ingroup turb_model
 *
 *  Implementation of laminar flow model
 */
template <typename Transport>
class Laminar : public TurbModel<Transport>
{
public:
    static std::string identifier()
    {
        return "Laminar-" + Transport::identifier();
    }

    explicit Laminar(CFDSim& sim) : TurbModel<Transport>(sim) {}

    virtual ~Laminar() = default;

    //! String representation of this model for I/O
    virtual std::string model_name() const override { return "laminar"; }

    //! Update the effective/turbulent viscosity field
    virtual void update_turbulent_viscosity(const FieldState fstate) override;

    //! No post advance work for this model
    virtual void post_advance_work() override {}

    //! Return the turbulent viscosity field (just the same as the effective
    //! field)
    virtual Field& mu_turb() override { return this->mueff(); }

    //! Return the thermal diffusivity field (just the same as effective field)
    virtual Field& alpha_turb() override { return this->alphaeff(); }

    //! Indicate that this model is not a turbulent model type
    virtual bool is_turbulent() const override { return false; }

    //! Interface to update effective viscosity (mu_eff = mu + mu_t)
    virtual void update_mueff(Field& mueff) override;

    //! Interface to update effective thermal diffusivity
    virtual void update_alphaeff(Field& alphaeff) override;

    //! Interface to update scalar diffusivity based on Schmidt number
    virtual void
    update_scalar_diff(Field& deff, const std::string& name) override;

    //! Return model coefficients dictionary
    virtual TurbulenceModel::CoeffsDictType model_coeffs() const override;
};

} // namespace turbulence
} // namespace amr_wind

#endif /* LAMINARMODEL_H */

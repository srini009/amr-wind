#include "amr-wind/equation_systems/DiffusionOps.H"
#include "amr-wind/utilities/console_io.H"

#include "AMReX_MLTensorOp.H"

namespace amr_wind {
namespace pde {

template <typename LinOp>
DiffSolverIface<LinOp>::DiffSolverIface(
    PDEFields& fields, const bool has_overset, const std::string& prefix)
    : m_pdefields(fields)
    , m_density(fields.repo.get_field("density"))
    , m_options(prefix, m_pdefields.field.name() + "_" + prefix)
{
    amrex::LPInfo isolve = m_options.lpinfo();
    amrex::LPInfo iapply;

    iapply.setMaxCoarseningLevel(0);

    const auto& mesh = m_pdefields.repo.mesh();
    if (!has_overset) {
        m_solver.reset(new LinOp(
            mesh.Geom(0, mesh.finestLevel()),
            mesh.boxArray(0, mesh.finestLevel()),
            mesh.DistributionMap(0, mesh.finestLevel()), isolve));
        m_applier.reset(new LinOp(
            mesh.Geom(0, mesh.finestLevel()),
            mesh.boxArray(0, mesh.finestLevel()),
            mesh.DistributionMap(0, mesh.finestLevel()), iapply));
    } else {
        auto imask = fields.repo.get_int_field("mask_cell").vec_const_ptrs();
        m_solver.reset(new LinOp(
            mesh.Geom(0, mesh.finestLevel()),
            mesh.boxArray(0, mesh.finestLevel()),
            mesh.DistributionMap(0, mesh.finestLevel()), imask, isolve));
        m_applier.reset(new LinOp(
            mesh.Geom(0, mesh.finestLevel()),
            mesh.boxArray(0, mesh.finestLevel()),
            mesh.DistributionMap(0, mesh.finestLevel()), imask, iapply));
    }

    m_solver->setMaxOrder(m_options.max_order);
    m_applier->setMaxOrder(m_options.max_order);

    // It is the sub-classes responsibility to set the linear solver BC for the
    // operators.
}

template <typename LinOp>
void DiffSolverIface<LinOp>::setup_operator(
    LinOp& linop,
    const amrex::Real alpha,
    const amrex::Real beta,
    const FieldState fstate)
{
    BL_PROFILE("amr-wind::setup_operator");
    auto& repo = m_pdefields.repo;
    const int nlevels = repo.num_active_levels();

    linop.setScalars(alpha, beta);
    for (int lev = 0; lev < nlevels; ++lev) {
        linop.setLevelBC(lev, &m_pdefields.field(lev));
    }
    this->set_acoeffs(linop, fstate);
    set_bcoeffs(linop);
}

template <typename LinOp>
void DiffSolverIface<LinOp>::set_acoeffs(LinOp& linop, const FieldState fstate)
{
    BL_PROFILE("amr-wind::set_acoeffs");
    auto& repo = m_pdefields.repo;
    const int nlevels = repo.num_active_levels();
    const auto& density = m_density.state(fstate);

    for (int lev = 0; lev < nlevels; ++lev) {
        linop.setACoeffs(lev, density(lev));
    }
}

template <typename LinOp>
void DiffSolverIface<LinOp>::setup_solver(amrex::MLMG& mlmg)
{
    BL_PROFILE("amr-wind::setup_solver");
    // Set all MLMG options
    m_options(mlmg);
}

template <typename LinOp>
void DiffSolverIface<LinOp>::linsys_solve_impl()
{
    FieldState fstate = FieldState::New;
    auto& repo = this->m_pdefields.repo;
    auto& field = this->m_pdefields.field;
    const auto& density = m_density.state(fstate);
    const int nlevels = repo.num_active_levels();
    const int ndim = field.num_comp();
    auto rhs_ptr = repo.create_scratch_field("rhs", field.num_comp(), 0);

    // Always multiply with rho since there is no diffusion term for density
    for (int lev = 0; lev < nlevels; ++lev) {
        auto& rhs = (*rhs_ptr)(lev);

#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(rhs, amrex::TilingIfNotGPU()); mfi.isValid();
             ++mfi) {
            const auto& bx = mfi.tilebox();
            const auto& rhs_a = rhs.array(mfi);
            const auto& fld = field(lev).const_array(mfi);
            const auto& rho = density(lev).const_array(mfi);

            amrex::ParallelFor(
                bx, ndim,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
                    rhs_a(i, j, k, n) = rho(i, j, k) * fld(i, j, k, n);
                });
        }
    }

    amrex::MLMG mlmg(*this->m_solver);
    this->setup_solver(mlmg);

    mlmg.solve(
        field.vec_ptrs(), rhs_ptr->vec_const_ptrs(), this->m_options.rel_tol,
        this->m_options.abs_tol);

    io::print_mlmg_info(field.name() + "_solve", mlmg);
}

template <typename LinOp>
void DiffSolverIface<LinOp>::linsys_solve(const amrex::Real dt)
{
    FieldState fstate = FieldState::New;
    this->setup_operator(*this->m_solver, 1.0, dt, fstate);
    this->linsys_solve_impl();
}

template class DiffSolverIface<amrex::MLABecLaplacian>;
template class DiffSolverIface<amrex::MLTensorOp>;

} // namespace pde
} // namespace amr_wind

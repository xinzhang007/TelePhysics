"""Shared physics constants and helpers for TelePhysics pipeline.

Centralises material definitions, force-field construction, surface
defaults, and the simulation default config so that both
``pipeline/simulation.py`` and ``pipeline/auto_config.py`` import from
one place.
"""

import genesis as gs

# ── Default simulation config ────────────────────────────────────────────
DEFAULT_CONFIG = {
    "simulation": {
        "max_retries": 8,
        "dist_thresh": 0.01,
        "cos_thresh": 0.8,
        "low_p": 5,
        "n_steps": 300,
        "fps": 60,
        "camera_mv": 0,
        "camera_tre": 0,
    },
    "objects": {},
}

# ── Material catalogue with default parameters ──────────────────────────
MATERIAL_DEFAULTS = {
    # Rigid
    "rigid": {"rho": 200.0},
    # MPM
    "mpm_elastic": {"E": 3e5, "nu": 0.2, "rho": 1000.0, "model": "corotation"},
    "mpm_elastoplastic": {
        "E": 3e4, "nu": 0.4, "rho": 100.0,
        "use_von_mises": True, "von_mises_yield_stress": 10000.0,
    },
    "mpm_sand": {"E": 5e5, "nu": 0.2, "rho": 1800, "friction_angle": 45},
    "mpm_liquid": {"E": 1e6, "nu": 0.2, "rho": 1000.0, "viscous": False},
    "mpm_snow": {"E": 1e6, "nu": 0.2, "rho": 1000.0, "yield_lower": 2.5e-2, "yield_higher": 4.5e-3},
    "mpm_muscle": {"E": 1e6, "nu": 0.2, "rho": 1000.0, "model": "neohooken", "n_groups": 1},
    # PBD
    "pbd_elastic": {
        "rho": 1000.0, "static_friction": 0.15, "kinetic_friction": 0.15,
        "stretch_compliance": 0.0, "bending_compliance": 0.0,
        "volume_compliance": 0.0, "stretch_relaxation": 0.1,
        "bending_relaxation": 0.1, "volume_relaxation": 0.1,
    },
    "pbd_cloth": {
        "rho": 4.0, "static_friction": 0.15, "kinetic_friction": 0.15,
        "stretch_compliance": 1e-7, "bending_compliance": 1e-5,
        "stretch_relaxation": 0.3, "bending_relaxation": 0.1, "air_resistance": 1e-3,
    },
    "pbd_liquid": {"rho": 1000.0, "density_relaxation": 0.2, "viscosity_relaxation": 0.01},
    "pbd_particle": {"rho": 1000.0},
}

VALID_MATERIALS = list(MATERIAL_DEFAULTS.keys())

# ── Default surface colours for particle-type materials ──────────────────
SURFACE_DEFAULTS = {
    "mpm_sand":           [0.76, 0.70, 0.50],
    "mpm_liquid":         [0.20, 0.50, 0.90],
    "mpm_snow":           [0.95, 0.97, 1.00],
    "pbd_liquid":         [0.20, 0.50, 0.90],
    "pbd_particle":       [0.70, 0.70, 0.70],
    "sph_liquid":         [0.20, 0.50, 0.90],
}

VALID_FORCE_TYPES = {"constant", "wind", "point", "drag", "noise", "vortex", "turbulence"}


# ── Material factory ─────────────────────────────────────────────────────

def get_material(mat_type, params=None):
    """Return a Genesis material instance for *mat_type*."""
    if params is None:
        params = {}

    # ── Rigid ──
    if mat_type == "rigid":
        return gs.materials.Rigid(
            rho=float(params.get("rho", 200.0)),
            friction=float(params["friction"]) if "friction" in params else None,
        )

    # ── MPM Elastic ──
    elif mat_type == "mpm_elastic":
        return gs.materials.MPM.Elastic(
            E=float(params.get("E", 3e5)),
            nu=float(params.get("nu", 0.2)),
            rho=float(params.get("rho", 1000.0)),
            model=str(params.get("model", "corotation")),
        )

    # ── MPM ElastoPlastic ──
    elif mat_type == "mpm_elastoplastic":
        return gs.materials.MPM.ElastoPlastic(
            E=float(params.get("E", 3e4)),
            nu=float(params.get("nu", 0.4)),
            rho=float(params.get("rho", 100.0)),
            use_von_mises=bool(params.get("use_von_mises", True)),
            von_mises_yield_stress=float(params.get("von_mises_yield_stress", 10000.0)),
        )

    # ── MPM Sand ──
    elif mat_type == "mpm_sand":
        return gs.materials.MPM.Sand(
            E=float(params.get("E", 5e5)),
            nu=float(params.get("nu", 0.2)),
            rho=float(params.get("rho", 1800)),
            friction_angle=float(params.get("friction_angle", 45)),
        )

    # ── MPM Liquid ──
    elif mat_type == "mpm_liquid":
        return gs.materials.MPM.Liquid(
            E=float(params.get("E", 1e6)),
            nu=float(params.get("nu", 0.2)),
            rho=float(params.get("rho", 1000.0)),
            viscous=bool(params.get("viscous", False)),
        )

    # ── MPM Snow ──
    elif mat_type == "mpm_snow":
        return gs.materials.MPM.Snow(
            E=float(params.get("E", 1e6)),
            nu=float(params.get("nu", 0.2)),
            rho=float(params.get("rho", 1000.0)),
            yield_lower=float(params.get("yield_lower", 2.5e-2)),
            yield_higher=float(params.get("yield_higher", 4.5e-3)),
        )

    # ── MPM Muscle ──
    elif mat_type == "mpm_muscle":
        return gs.materials.MPM.Muscle(
            E=float(params.get("E", 1e6)),
            nu=float(params.get("nu", 0.2)),
            rho=float(params.get("rho", 1000.0)),
            model=str(params.get("model", "neohooken")),
            n_groups=int(params.get("n_groups", 1)),
        )

    # ── PBD Elastic (3D soft body) ──
    elif mat_type == "pbd_elastic":
        return gs.materials.PBD.Elastic(
            rho=float(params.get("rho", 1000.0)),
            static_friction=float(params.get("static_friction", 0.15)),
            kinetic_friction=float(params.get("kinetic_friction", 0.15)),
            stretch_compliance=float(params.get("stretch_compliance", 0.0)),
            bending_compliance=float(params.get("bending_compliance", 0.0)),
            volume_compliance=float(params.get("volume_compliance", 0.0)),
            stretch_relaxation=float(params.get("stretch_relaxation", 0.1)),
            bending_relaxation=float(params.get("bending_relaxation", 0.1)),
            volume_relaxation=float(params.get("volume_relaxation", 0.1)),
        )

    # ── PBD Cloth (2D thin sheet) ──
    elif mat_type == "pbd_cloth":
        return gs.materials.PBD.Cloth(
            rho=float(params.get("rho", 4.0)),
            static_friction=float(params.get("static_friction", 0.15)),
            kinetic_friction=float(params.get("kinetic_friction", 0.15)),
            stretch_compliance=float(params.get("stretch_compliance", 1e-7)),
            bending_compliance=float(params.get("bending_compliance", 1e-5)),
            stretch_relaxation=float(params.get("stretch_relaxation", 0.3)),
            bending_relaxation=float(params.get("bending_relaxation", 0.1)),
            air_resistance=float(params.get("air_resistance", 1e-3)),
        )

    # ── PBD Liquid ──
    elif mat_type == "pbd_liquid":
        return gs.materials.PBD.Liquid(
            rho=float(params.get("rho", 1000.0)),
            density_relaxation=float(params.get("density_relaxation", 0.2)),
            viscosity_relaxation=float(params.get("viscosity_relaxation", 0.01)),
        )

    # ── PBD Particle ──
    elif mat_type == "pbd_particle":
        return gs.materials.PBD.Particle(
            rho=float(params.get("rho", 1000.0)),
        )

    else:
        print(f"[WARN] Unknown material type '{mat_type}', falling back to MPM.Elastic")
        return gs.materials.MPM.Elastic()


# ── Force-field factory ──────────────────────────────────────────────────

def get_force_field(force_cfg):
    """Map a force config dict to a ``gs.force_fields.*`` instance."""
    ftype = force_cfg.get("type", "constant")

    if ftype == "constant":
        return gs.force_fields.Constant(
            direction=tuple(force_cfg.get("direction", [0, 0, -1])),
            strength=float(force_cfg.get("strength", 9.8)),
        )
    elif ftype == "wind":
        return gs.force_fields.Wind(
            direction=tuple(force_cfg.get("direction", [1, 0, 0])),
            strength=float(force_cfg.get("strength", 1.0)),
            radius=float(force_cfg.get("radius", 1.0)),
            center=tuple(force_cfg.get("center", [0, 0, 0])),
        )
    elif ftype == "point":
        return gs.force_fields.Point(
            strength=float(force_cfg.get("strength", 1.0)),
            position=tuple(force_cfg.get("position", [0, 0, 0])),
            falloff_pow=float(force_cfg.get("falloff_pow", 0.0)),
            flow=float(force_cfg.get("flow", 1.0)),
        )
    elif ftype == "drag":
        return gs.force_fields.Drag(
            linear=float(force_cfg.get("linear", 0.0)),
            quadratic=float(force_cfg.get("quadratic", 0.0)),
        )
    elif ftype == "noise":
        return gs.force_fields.Noise(
            strength=float(force_cfg.get("strength", 1.0)),
        )
    elif ftype == "vortex":
        return gs.force_fields.Vortex(
            direction=tuple(force_cfg.get("direction", [0, 0, 1])),
            center=tuple(force_cfg.get("center", [0, 0, 0])),
            strength_perpendicular=float(force_cfg.get("strength_perpendicular", 20.0)),
            strength_radial=float(force_cfg.get("strength_radial", 0.0)),
            falloff_pow=float(force_cfg.get("falloff_pow", 2.0)),
            falloff_min=float(force_cfg.get("falloff_min", 0.01)),
            falloff_max=float(force_cfg.get("falloff_max", float("inf"))),
            damping=float(force_cfg.get("damping", 0.0)),
        )
    elif ftype == "turbulence":
        return gs.force_fields.Turbulence(
            strength=float(force_cfg.get("strength", 1.0)),
            frequency=float(force_cfg.get("frequency", 3)),
            flow=float(force_cfg.get("flow", 0.0)),
            seed=force_cfg.get("seed", None),
        )
    else:
        raise ValueError(f"Unknown force field type: {ftype}")

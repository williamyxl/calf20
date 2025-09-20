import io
import os
import numpy as np
from ase import io as aseio
from ase import build as asebuild
from ase.optimize import LBFGS
from ase.filters import FrechetCellFilter
from fairchem.core import pretrained_mlip, FAIRChemCalculator


max_steps = 500
traj_fpath = os.path.expanduser("/scratch/group/p.cis240698.000/xyan11/workdir/calf-20/uma/squcalf-20/traj.extxyz")
cif_fpath = os.path.expanduser("/scratch/group/p.cis240698.000/xyan11/workdir/calf-20/uma/squcalf-20/squcalf-20.cif")
try:
    if os.path.isfile(traj_fpath):
        os.remove(traj_fpath)
except PermissionError:
    traj_fpath = os.path.expanduser(traj_fpath)

eV2J = 1.60217663e-19
m2Ang = 1e10
atm2eV_per_Ang3 = 101325 / (m2Ang**3) / eV2J
cp2k_max_force_Ha_per_Bohr = 4.5E-4
_Ha_per_Bohr_2_eV_per_Ang = 51.4221
fmax_max_force_eV_per_Ang = cp2k_max_force_Ha_per_Bohr * _Ha_per_Bohr_2_eV_per_Ang

predictor = pretrained_mlip.get_predict_unit("uma-m-1p1", device="cuda", cache_dir="/scratch/group/p.cis240698.000/xyan11/workdir/calf-20/uma/uma-cache")
# oc20: use this for catalysis
# omat: use this for inorganic materials
# omol: use this for molecules
# odac: use this for MOFs
# omc: use this for molecular crystals

atoms = aseio.read(cif_fpath, format="cif")
supercell222 = asebuild.make_supercell(atoms, np.array([[2,0,0], [0,2,0], [0,0,2]]))

supercell222.calc = FAIRChemCalculator(predictor, task_name="odac")
fcf = FrechetCellFilter(supercell222, scalar_pressure=1*atm2eV_per_Ang3)
opt = LBFGS(fcf)
ase_traj_buffer = io.BytesIO()
traj = aseio.Trajectory(ase_traj_buffer, 'w', atoms=supercell222)
opt.attach(traj)
opt.run(fmax=fmax_max_force_eV_per_Ang, steps=max_steps)

trajectory_atoms = list(aseio.read(ase_traj_buffer, format='traj', index=':'))
traj_extxyz_buffer = io.StringIO()
aseio.write(traj_extxyz_buffer, trajectory_atoms, format="extxyz")
traj_extxyz_str = traj_extxyz_buffer.getvalue()

with io.open(traj_fpath, "w", newline="\n") as wf:
    wf.write(traj_extxyz_str)

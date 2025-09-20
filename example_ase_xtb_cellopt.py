import io
import os
import numpy as np
from ase import io as aseio
from ase import build as asebuild
from ase.optimize import LBFGS
from ase.filters import FrechetCellFilter
from tblite.ase import TBLite


os.environ['OMP_STACKSIZE'] = '4.9G'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_MAX_ACTIVE_LEVELS'] = '1'

max_steps = 500
traj_fpath = "traj.extxyz"
structure_fpath = "../initial.extxyz"

eV2J = 1.60217663e-19
m2Ang = 1e10
atm2eV_per_Ang3 = 101325 / (m2Ang**3) / eV2J
cp2k_max_force_Ha_per_Bohr = 2.0E-4
_Ha_per_Bohr_2_eV_per_Ang = 51.4221
fmax_max_force_eV_per_Ang = cp2k_max_force_Ha_per_Bohr * _Ha_per_Bohr_2_eV_per_Ang

atoms = aseio.read(structure_fpath, format="extxyz")
supercell222 = asebuild.make_supercell(atoms, np.array([[2,0,0], [0,2,0], [0,0,2]]))

supercell222.calc = TBLite(method="GFN1-xTB", max_iterations=1000, verbosity=0)
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

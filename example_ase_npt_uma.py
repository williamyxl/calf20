import io
import os
import numpy as np
import pandas as pd

import ase
from ase import md
from ase import units
from ase import io as aseio
from ase.md import nose_hoover_chain
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from fairchem.core import pretrained_mlip, FAIRChemCalculator


Faraday_C_per_mol = 96485.3321
Avogadro = 6.02214076e23
eV_per_J = Avogadro / Faraday_C_per_mol
Ang_per_m = 1e10
eV_per_Pa_Ang3 = eV_per_J / ((Ang_per_m)**3)

predictor = pretrained_mlip.get_predict_unit(
    #"uma-m-1p1",
    "uma-s-1p1",
    device="cuda",
    cache_dir="/work/nvme/bcbw/xyan11/workdir/calf20/uma-model-cache"  # DeltaAI
    # cache_dir="/scratch/group/p.cis240698.000/xyan11/workdir/calf-20/uma/uma-cache"  # TAMU ACES
    # cache_dir="/mnt/f/workdir/uma-cache"  # 7950x Titan V
    # cache_dir="/mnt/d/workdir/uma-cache"  # 7950x 4090
)


def thermo(atoms, step, log_fpath='./thermo.log'):
    pxx, pyy, pzz, pyz, pxz, pxy = atoms.get_stress()
    T = atoms.get_temperature()
    V = atoms.get_volume()
    M = atoms.cell
    with io.open(log_fpath, "a", newline="\n") as wf:
        wf.write(
            '%d' % step + ',' + 
            "%.10f" % pxx + "," + 
            "%.10f" % pyy + "," + 
            "%.10f" % pzz + "," + 
            "%.10f" % pyz + "," + 
            "%.10f" % pxz + "," + 
            "%.10f" % pxy + "," + 
            "%.10f" % T + "," + 
            "%.10f" % V + "," + 
            "%.10f" % M[0,0] + "," + 
            "%.10f" % M[0,1] + "," + 
            "%.10f" % M[0,2] + "," + 
            "%.10f" % M[1,0] + "," + 
            "%.10f" % M[1,1] + "," + 
            "%.10f" % M[1,2] + "," + 
            "%.10f" % M[2,0] + "," + 
            "%.10f" % M[2,1] + "," + 
            "%.10f" % M[2,2] + "," + 
            "%.10f" % np.linalg.norm(M[0]) + "," + 
            "%.10f" % np.linalg.norm(M[1]) + "," + 
            "%.10f" % np.linalg.norm(M[2]) + "," + 
            "%.10f" % (180 / np.pi * np.arccos(np.dot(M[1], M[2]) / np.linalg.norm(M[1]) / np.linalg.norm(M[2]))) + "," + 
            "%.10f" % (180 / np.pi * np.arccos(np.dot(M[0], M[2]) / np.linalg.norm(M[0]) / np.linalg.norm(M[2]))) + "," + 
            "%.10f" % (180 / np.pi * np.arccos(np.dot(M[0], M[1]) / np.linalg.norm(M[0]) / np.linalg.norm(M[1]))) + "\n"
        )
    print("Stress: pxx=" + "%.10f" % pxx + ", pyy=" + "%.10f" % pyy + ", pzz=" + "%.10f" % pzz + ", pyz=" + "%.10f" % pyz + ", pxz=" + "%.10f" % pxz + ", pxy=" + "%.10f" % pxy, flush=True)
    print("Temperature: " + "%.10f" % T + " K", flush=True)
    print("Volume: " + "%.10f" % V + " Ang^3", flush=True)
    print("Vector A: " + np.array2string(M[0]), flush=True)
    print("Vector B: " + np.array2string(M[1]), flush=True)
    print("Vector C: " + np.array2string(M[2]), flush=True)
    print("A: " + "%.10f" % (np.linalg.norm(M[0])), flush=True)
    print("B: " + "%.10f" % (np.linalg.norm(M[1])), flush=True)
    print("C: " + "%.10f" % (np.linalg.norm(M[2])), flush=True)
    print("Alpha: " + "%.10f" % (180 / np.pi * np.arccos(np.dot(M[1], M[2]) / np.linalg.norm(M[1]) / np.linalg.norm(M[2]))), flush=True)
    print("Beta: " + "%.10f" % (180 / np.pi * np.arccos(np.dot(M[0], M[2]) / np.linalg.norm(M[0]) / np.linalg.norm(M[2]))), flush=True)
    print("Gamma: " + "%.10f" % (180 / np.pi * np.arccos(np.dot(M[0], M[1]) / np.linalg.norm(M[0]) / np.linalg.norm(M[1]))), flush=True)


def write_xyzfile_to_string(atom_df, box_vec=None, origin=[0,0,0]):
    if type(box_vec) == type(None):
        box_str = ""
    else:
        box_str = 'Lattice="' + ' '.join(map(str, box_vec.flatten())) + '" Origin="' + ' '.join(map(str, origin)) + '" Properties=species:S:1:pos:R:3'
    return '%d' % len(atom_df.index) + "\n" + box_str + "\n" + atom_df[["el", "x", "y", "z"]].to_string(header=None, index=None) + "\n"


def write_ase_to_extxyz_str(atoms):
    df = pd.DataFrame(atoms.positions, columns=["x", "y", "z"])
    df["el"] = atoms.get_chemical_symbols()
    return write_xyzfile_to_string(df[["el", "x", "y", "z"]], box_vec=atoms.cell)


total_steps = 4000000
thermo_freq = 1000
traj_str = ""
traj_fpath = "traj.extxyz"
log_fpath = "thermo.log"
temperature = 300

atoms = aseio.read("../lmp.data", format="lammps-data", units="real", atom_style="charge")
atoms.calc = FAIRChemCalculator(predictor, task_name="odac")
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

dyn = nose_hoover_chain.MTKNPT(
    atoms,
    timestep=1.0 * units.fs,
    temperature_K=temperature,
    pressure_au=1.01325 * units.bar,
    tdamp=100 * units.fs,
    pdamp=1000 * units.fs,
)

# write log file header
with io.open(log_fpath, "a", newline="\n") as wf:
    wf.write("Step,pxx,pyy,pzz,pyz,pxz,pxy,T,V,Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz,A,B,C,alpha,beta,gamma\n")

for run_i in range(0, int(total_steps / thermo_freq)):
    step = run_i * thermo_freq
    # save frame
    thermo(atoms, step, log_fpath)
    with io.open(traj_fpath, "a", newline="\n") as wf:
        wf.write(write_ase_to_extxyz_str(atoms))

    # run dynamics
    dyn.run(thermo_freq)

# save final frame
thermo(atoms, step, log_fpath)
with io.open(traj_fpath, "a", newline="\n") as wf:
    wf.write(write_ase_to_extxyz_str(atoms))

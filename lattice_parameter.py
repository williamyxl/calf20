import os
import io
import numpy as np
import pandas as pd


def transformation_matrix_from_lattice_parameters(a, b, c, alpha, beta, gamma):
    cosal = np.cos(alpha * np.pi / 180)
    sinal = np.sin(alpha * np.pi / 180)
    cosbe = np.cos(beta * np.pi / 180)
    sinbe = np.sin(beta * np.pi / 180)
    cosga = np.cos(gamma * np.pi / 180)
    singa = np.sin(gamma * np.pi / 180)
    n2 = (cosal - (cosga * cosbe)) / (singa)
    M = np.array([[a, 0, 0],
                  [b*cosga, b*singa, 0],
                  [c*cosbe, c*n2, c*np.sqrt((sinbe*sinbe) - (n2*n2))]])
    return M

def read_ciffile(fpath):
    if os.path.isfile(fpath):
        read_str = None
        with io.open(fpath, "r", newline="\n") as rf:
            read_str = rf.read()
    else:
        read_str = fpath
    segments = read_str.split("loop_\n")
    HM_sym = None
    lat_param = {"a": None, "b": None, "c": None, "alpha": None, "beta": None, "gamma": None}
    for seg in segments:
        if "_symmetry_space_group_name_H-M" in seg:
            HM_sym = seg.split("_symmetry_space_group_name_H-M")[1].split("\n")[0].strip()
        if "_cell_length_a" in seg:
            for line in seg.strip().split("\n"):
                if "_cell_length_a" in line:
                    lat_param["a"] = float(line.split("_cell_length_a")[1])
                elif "_cell_length_b" in line:
                    lat_param["b"] = float(line.split("_cell_length_b")[1])
                elif "_cell_length_c" in line:
                    lat_param["c"] = float(line.split("_cell_length_c")[1])
                elif "_cell_angle_alpha" in line:
                    lat_param["alpha"] = float(line.split("_cell_angle_alpha")[1])
                elif "_cell_angle_beta" in line:
                    lat_param["beta"] = float(line.split("_cell_angle_beta")[1])
                elif "_cell_angle_gamma" in line:
                    lat_param["gamma"] = float(line.split("_cell_angle_gamma")[1])
        elif "_atom_site_fract_x" in seg:
            atom_lines = seg.strip().split("\n")
            header = []
            data_line_i = None
            for line_i in range(0, len(atom_lines)):
                line = atom_lines[line_i]
                if "_atom_" in line:
                    header.append(line.strip())
                else:
                    data_line_i = line_i
                    break
    df = pd.read_csv(io.StringIO("\n".join(atom_lines[data_line_i:])), sep="\s+", names=header)
    if "_atom_site_type_symbol" in df.columns:
        df["el"] = df["_atom_site_type_symbol"]
    elif "_atom_site_label" in df.columns:
        df["el"] = df["_atom_site_label"]
    df["xs"] = df["_atom_site_fract_x"]
    df["ys"] = df["_atom_site_fract_y"]
    df["zs"] = df["_atom_site_fract_z"]
    df["q"] = df["_atom_site_charge"]
    df = df[["el", "xs", "ys", "zs", "q"]]
    M = transformation_matrix_from_lattice_parameters(lat_param["a"], lat_param["b"], lat_param["c"], lat_param["alpha"], lat_param["beta"], lat_param["gamma"])
    return lat_param, M, df

lat_params = []
for x in sorted(os.listdir()):
    if x.endswith("_DDEC6.cif"):
        lat_params.append(read_ciffile(x)[0])
        lat_params[-1]["structure"] = x
ldf = pd.DataFrame(lat_params)
ldf.loc[:, ["a", "b", "c"]] = ldf.loc[:, ["a", "b", "c"]].values / 2
ldf

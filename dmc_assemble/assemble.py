# dependents: numpy, pandas, scipy only
# no need for ASE, PyMatGen, etc.
# pure coordinate manipulation
# author: Xiaoli Yan, williamyxl@gmail.com
#

import io
import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as ScipyR


# helper function to read xyz or extxyz file, only one frame is returned if many frames exist
def read_one_xyz2df(xyzfpath, extxyz=False):
    read_str = None
    with io.open(xyzfpath, "r", newline="\n") as rf:
        read_str = rf.read()
    lines = read_str.split("\n")
    curr_line = lines[0]
    natoms = int(curr_line)
    comment = curr_line = lines[1]
    mol_str = lines[2:2+natoms]
    if extxyz:
        #header =  # lines[1].split(" ")[0]
        if '"' in comment:
            items = list(filter(None, comment.split('"')))
        elif "'" in comment:
            items = list(filter(None, comment.split("'")))
        else:
            print("not a valid extxyz file comment")
            #return None
            pass
        for x_i in range(0, len(items)):
            if items[x_i] == "Lattice=":
                M = np.loadtxt(io.StringIO(items[x_i+1])).reshape(3, 3)
                break
        #df = pd.read_csv(io.StringIO("\n".join(mol_str)), header=None, sep=r"\s+", names=["el", "x", "y", "z", "Fx", "Fy", "Fz", "q"])
        df = pd.read_csv(io.StringIO("\n".join(mol_str)), header=None, sep=r"\s+", names=["el", "x", "y", "z"])
        return M, df
    else:
        df = pd.read_csv(io.StringIO("\n".join(mol_str)), header=None, sep=r"\s+", names=["el", "x", "y", "z"])
        return df


# rotate a group of atoms (selected or all) around a given 3D vector, with given stationary point and rotation angle theta
# need "atom_id" column, 0-based or 1-based, theta is in degree
def rotate_selected_around_vector(atom_df, stationary_point, selected_atom_ids, rotation_axis_vector, theta):
    selected_xyz = atom_df.loc[atom_df["atom_id"].isin(selected_atom_ids), ["x", "y", "z"]].values
    rotation_matrix = ScipyR.from_mrp(rotation_axis_vector * np.tan(theta * np.pi / 180. / 4.)).as_matrix()
    selected_rotation_xyz = (rotation_matrix @ (selected_xyz - stationary_point).T).T + stationary_point
    atom_df.loc[:, ["x", "y", "z"]] = atom_df.loc[:, ["x", "y", "z"]].values.astype(float)
    atom_df.loc[atom_df["atom_id"].isin(selected_atom_ids), ["x", "y", "z"]] = selected_rotation_xyz.astype(float)
    return atom_df


# rotate a group of atoms (selected or all) around a pair of atoms (usually bonds, given by atom ids), with rotation angle theta
# need "atom_id" column, 0-based or 1-based, theta is in degree
def rotate_selected_around_bond(atom_df, selected_atom_ids, rotation_axis_atom_ids, theta):
    anchor_atom_id = rotation_axis_atom_ids[0]
    anchor_xyz = atom_df.loc[atom_df["atom_id"].isin([anchor_atom_id]), ["x", "y", "z"]].values[0]
    rotation_axis_vector = (atom_df.loc[atom_df["atom_id"].isin([rotation_axis_atom_ids[0]]), ["x", "y", "z"]].values - \
                            atom_df.loc[atom_df["atom_id"].isin([rotation_axis_atom_ids[1]]), ["x", "y", "z"]].values).flatten()
    rotation_axis_vector = rotation_axis_vector / np.linalg.norm(rotation_axis_vector)
    selected_xyz = atom_df.loc[atom_df["atom_id"].isin(selected_atom_ids), ["x", "y", "z"]].values
    rotation_matrix = ScipyR.from_mrp(rotation_axis_vector * np.tan(theta * np.pi / 180. / 4.)).as_matrix()
    selected_rotation_xyz = (rotation_matrix @ (selected_xyz - anchor_xyz).T).T + anchor_xyz
    atom_df.loc[:, ["x", "y", "z"]] = atom_df.loc[:, ["x", "y", "z"]].values.astype(float)
    atom_df.loc[atom_df["atom_id"].isin(selected_atom_ids), ["x", "y", "z"]] = selected_rotation_xyz.astype(float)
    return atom_df


# rotate a group of atoms, so that the given orientation vector 1 (often a pair of atoms, or normal vector to a ring) aligns with another given orientation vector
# need "atom_id" column, 0-based or 1-based
def rotate_to_align_orientation_vector(atom_df, anchor_atom_id, orientation_vector1, orientation_vector2):
    anchor_xyz = atom_df.loc[atom_df["atom_id"].isin([anchor_atom_id]), ["x", "y", "z"]].values[0]
    orientation_vector1 = orientation_vector1 / np.linalg.norm(orientation_vector1)
    orientation_vector2 = orientation_vector2 / np.linalg.norm(orientation_vector2)
    axis = np.cross(orientation_vector1, orientation_vector2)
    axis = axis / np.linalg.norm(axis)
    theta = np.arccos(np.dot(orientation_vector1, orientation_vector2) / np.linalg.norm(orientation_vector1) / np.linalg.norm(orientation_vector2))
    selected_xyz = atom_df.loc[:, ["x", "y", "z"]].values
    rotation_matrix = ScipyR.from_mrp(axis * np.tan(theta / 4.)).as_matrix()
    selected_rotation_xyz = (rotation_matrix @ (selected_xyz - anchor_xyz).T).T + anchor_xyz
    atom_df.loc[:, ["x", "y", "z"]] = atom_df.loc[:, ["x", "y", "z"]].values.astype(float)
    atom_df.loc[:, ["x", "y", "z"]] = selected_rotation_xyz.astype(float) 
    return atom_df


# find the normal vector to a plane through p1, p2, p3
def find_normal_vector(p1, p2, p3):
    v1 = p1 - p2
    v2 = p1 - p3
    normal_vec = np.cross(v1, v2)
    normal_vec = normal_vec / np.linalg.norm(normal_vec)
    return normal_vec


# write CIF file, ASE cif writer might fail in VESTA
def write_ciffile(atom_df, M, fpath):
    if "xs" not in atom_df.columns and "x" in atom_df.columns:
        atom_df["xs"] = None
        atom_df["ys"] = None
        atom_df["zs"] = None
        atom_df.loc[:, ["xs", "ys", "zs"]] = atom_df.loc[:, ["x", "y", "z"]].values @ np.linalg.inv(M)
    atom_df["label"] = ""
    for el in atom_df["el"].unique():
        atom_df.loc[atom_df[atom_df["el"]==el].index, "label"] = [x+1 for x in range(0, atom_df["el"].value_counts()[el])]
    atom_df["label"] = atom_df["el"] + atom_df["label"].astype(str)
    atom_df["q"] = 0.
    atom_df = atom_df[["label", "el", "xs", "ys", "zs", "q"]]

    a = np.linalg.norm(M[0])
    b = np.linalg.norm(M[1])
    c = np.linalg.norm(M[2])
    alpha = 180 * np.arccos(np.dot(M[1], M[2]) / b / c) / np.pi
    beta = 180 * np.arccos(np.dot(M[0], M[2]) / a / c) / np.pi
    gamma = 180 * np.arccos(np.dot(M[0], M[1]) / a / b) / np.pi

    cifstring = """# P1 structure
_cell_length_a                           """ + "%.10f" % a + """
_cell_length_b                           """ + "%.10f" % b + """
_cell_length_c                           """ + "%.10f" % c + """
_cell_angle_alpha                        """ + "%.10f" % alpha + """
_cell_angle_beta                         """ + "%.10f" % beta + """
_cell_angle_gamma                        """ + "%.10f" % gamma + """

_symmetry_cell_setting          triclinic
_symmetry_space_group_name_Hall 'P 1'
_symmetry_space_group_name_H-M  'P 1'
_symmetry_Int_Tables_number     1

_symmetry_equiv_pos_as_xyz 'x,y,z'

loop_
   _atom_site_label
   _atom_site_type_symbol
   _atom_site_fract_x
   _atom_site_fract_y
   _atom_site_fract_z
   _atom_site_charge
""" + atom_df.loc[:, ["label", "el", "xs", "ys", "zs", "q"]].to_string(index=None, header=None, float_format=lambda x: "%.16f" % x)
    with io.open(fpath, "w", newline="\n") as wf:
        wf.write(cifstring)


# write a xyz or extxyz (if box is given) file
def write_xyzfile(atom_df, fpath, box_vec=None, origin=[0,0,0]):
    if type(box_vec) == type(None):
        box_str = ""
    else:
        box_str = 'Lattice="' + ' '.join(map(str, box_vec.flatten())) + '" Origin="' + ' '.join(map(str, origin)) + '" Properties=species:S:1:pos:R:3'
    with io.open(fpath, "w", newline="\n") as wf:
        wf.write("%d" % len(atom_df.index) + "\n" + box_str + "\n" + atom_df[["el", "x", "y", "z"]].to_string(header=None, index=None) + "\n")


# assembling dmc structure with prepared ligand xyz files
# triazole-like ligands: put the xyz files in tri_path
# 1. make sure all 3 Zn atoms are included and at the first 3 atoms in the xyz file
# 2. make sure the first atom is like the Zn atom as if it was connected to the tip position of triazole
# 3. make sure the second and third atoms are the Zn atoms as if they were connected to the bottom position of triazole
#                         Zn1
#                          |
#                          N
#                        /   \
#                 H -- C       C -- H
#                       \     /
#                        N - N
#                       /     \
#                     Zn2     Zn3
# xyz file content:
# <number of atoms>
# <comment line>
# Zn <Zn1 x coordinate> <Zn1 y coordinate> <Zn1 z coordinate>
# Zn <Zn2 x coordinate> <Zn2 y coordinate> <Zn2 z coordinate>
# Zn <Zn3 x coordinate> <Zn3 y coordinate> <Zn3 z coordinate>
# <remaining atoms>
# ...
#
# oxalate-like ligands: put the xyz files in oxa_fpath 
# make sure all 2 Zn atoms are included and at the first 2 atoms in the xyz file
#                O       O
#              /   \   /   \
#             /      C      \
#           Zn1      |       Zn2
#              \     C      /
#               \  /   \   /
#                O       O
# xyz file content:
# <number of atoms>
# <comment line>
# Zn <Zn1 x coordinate> <Zn1 y coordinate> <Zn1 z coordinate>
# Zn <Zn2 x coordinate> <Zn2 y coordinate> <Zn2 z coordinate>
# <remaining atoms>
# ...
def assemble_dmc(tri_fpath, oxa_fpath, assembled_fpath):
    # need Zn atoms up front and then nitrogen
    tri = read_one_xyz2df(tri_fpath)
    # need Zn atoms up front and then oxygen
    oxa = read_one_xyz2df(oxa_fpath)

    tri["role"] = "body"
    tri.at[0, "role"] = "tip"
    tri.at[1, "role"] = "base1"
    tri.at[2, "role"] = "base2"

    # align plane perpendicular to Z-axis
    tri["atom_id"] = tri.index
    axis_ids = tri[tri["role"].isin(["base1", "base2", "tip"])].index.tolist()
    normal_vector = find_normal_vector(tri.loc[axis_ids[0], ["x", "y", "z"]].values.astype(float),
                                       tri.loc[axis_ids[1], ["x", "y", "z"]].values.astype(float),
                                       tri.loc[axis_ids[2], ["x", "y", "z"]].values.astype(float))
    tri = rotate_to_align_orientation_vector(tri.copy(deep=True), 0, normal_vector, np.array([0, 0, 1]))

    base_axis = tri.loc[tri["role"]=="base1", ["x", "y", "z"]].values[0] - tri.loc[tri["role"]=="base2", ["x", "y", "z"]].values[0]
    tri = rotate_to_align_orientation_vector(tri.copy(deep=True), 0, base_axis, np.array([1, 0, 0]))
    tri.loc[:, ["x", "y", "z"]] = tri.loc[:, ["x", "y", "z"]].values - tri.loc[0, ["x", "y", "z"]].values

    # rotate 180 around normal vector of Zn-Zn-Zn plane
    tri2 = tri.copy(deep=True)
    tri2["atom_id"] = tri2.index
    axis_ids = tri2[tri2["role"].isin(["base1", "base2", "tip"])].index.tolist()
    normal_vector = find_normal_vector(tri.loc[axis_ids[0], ["x", "y", "z"]].values.astype(float),
                                       tri.loc[axis_ids[1], ["x", "y", "z"]].values.astype(float),
                                       tri.loc[axis_ids[2], ["x", "y", "z"]].values.astype(float))
    selected_ids = list(set(tri2.index.tolist()))
    stationary_point = (tri2.loc[tri2["role"]=="base1", ["x", "y", "z"]].values + tri2.loc[tri2["role"]=="base2", ["x", "y", "z"]].values) * 0.5
    tri2 = rotate_selected_around_vector(tri2.copy(deep=True), stationary_point, selected_ids, normal_vector, 180)
    tri2.at[0, "role"] = "tip2"
    double_tri = pd.concat([tri, tri2.loc[~tri2["role"].isin(["base1", "base2"]), ["el", "x", "y", "z", "role"]]], axis=0).reset_index(drop=True)
    tip1_xyz = double_tri.loc[double_tri["role"]=="tip", ["x", "y", "z"]].values.astype(float)[0]
    tip2_xyz = double_tri.loc[double_tri["role"]=="tip2", ["x", "y", "z"]].values.astype(float)[0]

    double_tri2 = double_tri.copy(deep=True)
    # rotate double tri -90 around Zn-Zn axis
    double_tri2["atom_id"] = double_tri2.index
    axis_ids = double_tri2[double_tri2["role"].isin(["base1", "base2"])].index.tolist()
    selected_ids = list(set(double_tri2.index.tolist()) - set(axis_ids))
    double_tri2 = rotate_selected_around_bond(double_tri2.copy(deep=True), selected_ids, axis_ids, 90)
    double_tri2_Zn = double_tri2[double_tri2["role"].isin(["base1", "base2"])]
    Zn_ids = double_tri2_Zn.index
    Zn1 = double_tri2_Zn.loc[Zn_ids[0], :]
    Zn2 = double_tri2_Zn.loc[Zn_ids[1], :]
    if Zn1["x"] > Zn2["x"]:
        anchor_Zn_id = Zn_ids[0]
        yz_id = Zn_ids[1]
    else:
        anchor_Zn_id = Zn_ids[1]
        yz_id = Zn_ids[0]
    disp = tip2_xyz - double_tri2.loc[anchor_Zn_id, ["x", "y", "z"]].values.astype(float)
    double_tri2.loc[:, ["x", "y", "z"]] = double_tri2.loc[:, ["x", "y", "z"]] + disp
    BC = double_tri2.loc[yz_id, ["x", "y", "z"]].values.astype(float)

    tri_frame = pd.concat([double_tri, double_tri2.loc[double_tri2["el"]!="Zn", :]], axis=0).reset_index(drop=True)

  
    oxa["role"] = "body"
    oxa.at[0, "role"] = "tip1"
    oxa.at[1, "role"] = "tip2"

    # align plane perpendicular to Y=Z plane
    oxa["atom_id"] = oxa.index
    axis_ids = [0, 1, 2]
    normal_vector = find_normal_vector(oxa.loc[axis_ids[0], ["x", "y", "z"]].values.astype(float),
                                       oxa.loc[axis_ids[1], ["x", "y", "z"]].values.astype(float),
                                       oxa.loc[axis_ids[2], ["x", "y", "z"]].values.astype(float))
    oxa = rotate_to_align_orientation_vector(oxa.copy(deep=True), 0, normal_vector, np.array([0, -1, np.sqrt(3)]))
    base_axis = oxa.loc[oxa["role"]=="tip2", ["x", "y", "z"]].values[0] - oxa.loc[oxa["role"]=="tip1", ["x", "y", "z"]].values[0]
    oxa = rotate_to_align_orientation_vector(oxa.copy(deep=True), 0, base_axis, np.array([1, 0, 0]))
    oxa.loc[:, ["x", "y", "z"]] = oxa.loc[:, ["x", "y", "z"]].values - oxa.loc[0, ["x", "y", "z"]].values

    # attach oxa to first double tri
    oxa["atom_id"] = oxa.index
    oxa2 = oxa.copy(deep=True)
    axis_ids = oxa2[oxa2["role"].isin(["tip1", "tip2"])].index.tolist()
    selected_ids = list(set(oxa2.index.tolist()) - set(axis_ids))
    oxa2 = rotate_selected_around_bond(oxa2.copy(deep=True), selected_ids, axis_ids, 90)

    oxa_Zn_ids = oxa[oxa["el"]=="Zn"].index.tolist()
    oxa_Zn1 = oxa.loc[oxa_Zn_ids[0], :]
    oxa_Zn2 = oxa.loc[oxa_Zn_ids[1], :]
    if oxa_Zn1["x"] < oxa_Zn2["x"]:
        anchor_Zn_id = oxa_Zn_ids[0]
    else:
        anchor_Zn_id = oxa_Zn_ids[1]
    oxa_anchor_xyz = oxa.loc[anchor_Zn_id, ["x", "y", "z"]].values.astype(float)

    tri_frame_Zn_ids = tri_frame[tri_frame["role"].isin(["base1", "base2"])].index.tolist()
    tri_frame_Zn1 = tri_frame.loc[tri_frame_Zn_ids[0], :]
    tri_frame_Zn2 = tri_frame.loc[tri_frame_Zn_ids[1], :]
    if tri_frame_Zn1["x"] > tri_frame_Zn2["x"]:
        anchor_Zn_id = tri_frame_Zn_ids[0]
        y_id = tri_frame_Zn_ids[1]
    else:
        anchor_Zn_id = tri_frame_Zn_ids[1]
        y_id = tri_frame_Zn_ids[0]
    disp = tri_frame.loc[anchor_Zn_id, ["x", "y", "z"]].values.astype(float) - oxa_anchor_xyz
    oxa.loc[:, ["x", "y", "z"]] = oxa.loc[:, ["x", "y", "z"]] + disp

    # attach oxa to second double tri
    oxa2 = rotate_selected_around_bond(oxa2.copy(deep=True), selected_ids, axis_ids, 60)
    oxa2_Zn_ids = oxa2[oxa2["el"]=="Zn"].index.tolist()
    oxa2_Zn1 = oxa2.loc[oxa2_Zn_ids[0], :]
    oxa2_Zn2 = oxa2.loc[oxa2_Zn_ids[1], :]
    if oxa2_Zn1["x"] < oxa2_Zn2["x"]:
        anchor_Zn_id = oxa2_Zn_ids[0]
        oxa2.at[oxa2_Zn_ids[1], "role"] = "xyz"
    else:
        anchor_Zn_id = oxa2_Zn_ids[1]
        oxa2.at[oxa2_Zn_ids[0], "role"] = "xyz"
    oxa2_anchor_xyz = oxa2.loc[anchor_Zn_id, ["x", "y", "z"]].values.astype(float)
    
    tri_frame_Zn_ids = tri_frame[tri_frame["role"].isin(["tip", "tip2"])].index.tolist()
    tri_frame_Zn1 = tri_frame.loc[tri_frame_Zn_ids[0], :]
    tri_frame_Zn2 = tri_frame.loc[tri_frame_Zn_ids[1], :]
    if tri_frame_Zn1["y"] > tri_frame_Zn2["y"]:
        anchor_Zn_id = tri_frame_Zn_ids[0]
    else:
        anchor_Zn_id = tri_frame_Zn_ids[1]
    disp = tri_frame.loc[anchor_Zn_id, ["x", "y", "z"]].values.astype(float) - oxa2_anchor_xyz
    oxa2.loc[:, ["x", "y", "z"]] = oxa2.loc[:, ["x", "y", "z"]] + disp

    total = pd.concat([tri_frame,
                       oxa.loc[list(set(oxa.index.tolist()) - set(oxa_Zn_ids))],
                       oxa2.loc[list(set(oxa2.index.tolist()) - set(oxa2_Zn_ids))]], axis=0).reset_index(drop=True)
    total["atom_id"] = total.index

    # lattice vectors
    ABC = oxa2.loc[oxa2["role"]=="xyz", ["x", "y", "z"]].values[0]
    B = (double_tri2.loc[double_tri2["role"]=="tip2", ["x", "y", "z"]].values.astype(float) - tri_frame.loc[y_id, ["x", "y", "z"]].values.astype(float)).flatten()
    A = (ABC - BC).flatten()
    C = (BC - B).flatten()
    M = np.array([A, B, C])

    # keeping lattice vector A stationary, and align lattice vectors B closer to Y axis
    A, B, C = M
    X = np.array([1, 0, 0])
    Y = np.array([0, 1, 0])
    Z = np.array([0, 0, 1])
    B_YZ = np.array([0, B[1], B[2]], dtype=float)
    theta = np.arccos(np.dot(B_YZ, Y) / np.linalg.norm(B_YZ) / np.linalg.norm(Y)) * 180 / np.pi

    origin = [0, 0, 0]
    fake_df = pd.DataFrame([origin] + M.tolist(), columns=["x", "y", "z"])
    fake_df["el"] = ["O", "A", "B", "C"]
    fake_df["atom_id"] = fake_df.index
    fake_df = fake_df[["el", "x", "y", "z", "atom_id"]]

    # rotate the lattice vectors
    fake_df2 = rotate_selected_around_vector(fake_df.copy(deep=True), np.array(origin), fake_df.index.tolist(), X, theta)
    M2 = fake_df2.loc[1:, ["x", "y", "z"]].values
    # rotate atoms
    total_df2 = rotate_selected_around_vector(total.copy(deep=True), np.array(origin), total.index.tolist(), X, theta)

    # if Y component of B is negative, redo rotation on opposite side
    if fake_df2.at[2, "y"] < fake_df2.at[2, "z"]:
        fake_df2 = rotate_selected_around_vector(fake_df.copy(deep=True), np.array(origin), fake_df.index.tolist(), -X, theta)
        M2 = fake_df2.loc[1:, ["x", "y", "z"]].values
        total_df2 = rotate_selected_around_vector(total.copy(deep=True), np.array(origin), total.index.tolist(), -X, theta)

    # make sure C vector has positive Z component
    if M2[2, 2] < 0:
        M2[2, :] = -M2[2, :]
        total_df2.loc[:, ["x", "y", "z"]] = total_df2.loc[:, ["x", "y", "z"]].values + M2[2, :]

    # write extxyz file for OVITO visualization
    # write_xyzfile(total_df2, assembled_fpath, box_vec=M2, origin=origin)

    return M2, total_df2


# already ignoring ASE functions, why not go further
def make_supercell(M, atom_df, N_supercell=[1, 1, 1]):
    # supercell
    elements = atom_df.loc[:, "el"].tolist()
    cart_coords = atom_df.loc[:, ["x", "y", "z"]].values
    frac_coords = cart_coords @ np.linalg.inv(M)
    
    # replicate unit cell
    all_df_list = []
    for na in range(0, N_supercell[0]):
        for nb in range(0, N_supercell[1]):
            for nc in range(0, N_supercell[2]):
                delta = np.array([na, nb, nc])
                x, y, z = ((frac_coords + delta) @ M).T.tolist()
                new_df = pd.DataFrame([elements, x, y, z]).T
                new_df.columns = ["el", "x", "y", "z"]
                all_df_list.append(new_df)
    atom_df = pd.concat(all_df_list, axis=0).reset_index(drop=True).astype({"el": str, "x": float, "y": float, "z": float})
    
    # scale up lattice cell
    M = np.diag(N_supercell) @ M

    # write an extxyz file for supercell for visualization
    #write_xyzfile(atom_df, os.path.join(folder_path, "supercell.extxyz"), box_vec=M, origin=[0, 0, 0])
    return M, atom_df



if __name__ == "__main__":
    for triangle_name in ["triazole", "246-4py-py", "4-carboxyl-pyrazole"]:
        for diamond_name in ["oxa", "squ", "bdc", "cub", "fum", "ttdc", "4,4'-bipy"]:
            triangle_fpath = os.path.join("tri", triangle_name + ".xyz")
            diamond_fpath = os.path.join("oxa", diamond_name + ".xyz")
            assembled_name = triangle_name + "_" + diamond_name
            assembled_fpath = os.path.join("assembled", assembled_name + ".extxyz")
            M, atom_df = assemble_dmc(triangle_fpath, diamond_fpath, assembled_fpath)
            write_ciffile(atom_df, M, os.path.join("assembled", assembled_name + ".cif"))

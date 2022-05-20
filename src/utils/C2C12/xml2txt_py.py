import xml.etree.ElementTree as ET
import numpy as np
import os


def AppendTracklet(cell, track_let, track_part):
    id = int(cell.get('id'))
    if len(cell) == 1:
        track_info = track_part.copy()
        track_info[0, 1] = id
        for info in cell.find('.//ss'):
            track_info[0, 0] = int(info.get('i'))
            track_info[0, 2] = int(float(info.get('x')))
            track_info[0, 3] = int(float(info.get('y')))
            track_let = np.append(track_let, track_info, axis=0)
        return track_let

    if len(cell) == 2:
        track_info = track_part.copy()
        track_info[0, 1] = id
        for info in cell.find('.//ss'):
            track_info[0, 0] = int(info.get('i'))
            track_info[0, 2] = int(float(info.get('x')))
            track_info[0, 3] = int(float(info.get('y')))
            track_let = np.append(track_let, track_info, axis=0)
        if len(cell.find('.//as')) == 2:
            for chil_cell in cell.find('.//as'):
                track_info = track_part.copy()
                track_info[0, -1] = id
                track_let = np.append(track_let,
                                      AppendTracklet(chil_cell, np.empty((0, 5)).astype('int32'), track_info), axis=0)
            return track_let
        else:
            return track_let

    else:
        print(f"else case: {cell}")
        return track_let

def main():
    # load xml files
    xml_folder = 'please/insert/path/to/XML_folder'
    save_path = os.path.join(xml_folder, "../Human_annotation_txt")
    os.makedirs(save_path, exist_ok=True)
    files = sorted([file for file in os.listdir(xml_folder) if file.endswith('xml')])
    for file in files:
        print(f"File handle: {file}")
        full_path = os.path.join(xml_folder, file)

        save_file_name = file.split('.')[-2].split(' ')[-2].split('_')[-1]
        if 'Full' in file:
            save_file_name = f"090303_exp1_{save_file_name}_GT_full.txt"
        else:
            save_file_name = f"090303_exp1_{save_file_name}_GT_semi.txt"
        full_save_path = os.path.join(save_path, save_file_name)
        tree = ET.parse(full_path)
        root = tree.getroot()
        num_files = len(root[0].findall("f"))
        print(f"Number of files : {num_files}")
        for i in range(len(root[0].findall("f"))):
            # number of cell
            num_cell = len(root[0][i][0].findall("a"))
            print(f"Find {num_cell} cells!!")
            track_let = np.empty((0, 5)).astype('int32')
            track_part = np.zeros((1, 5)).astype('int32')
            track_part[0, -1] = -1
            for par_cell in root[0][i][0].findall("a"):
                id = int(par_cell.get('id'))
                # print(id)
                track_let = AppendTracklet(par_cell, track_let, track_part.copy())

        np.savetxt(full_save_path,
                   track_let.astype('int'), fmt="%03d",
                   delimiter=" ")

if __name__ =="__main__":
    main()

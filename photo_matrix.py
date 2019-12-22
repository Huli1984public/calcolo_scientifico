import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob


def compare_rows(rows, data):
    ceil = max(rows)
    try:
        data[:, :] += my_filter.filter_force
        data[:, :][data[:, :] > ceil] = ceil
    except IndexError:
        data[:] += my_filter.filter_force
        data[:][data[:] > ceil] = ceil
    return data


class ImageFilter:

    def __init__(self, threshold=150, filter_force=20):
        self.stack = None

        # set how much filter is applied
        self.filter_force = filter_force
        # set filter threshold
        self.threshold = threshold

    def generate_test(self):
        data = None
        for i in range(0, 2):
            print(i, "i")
            random_matrix = np.random.randint(255, size=(4, 4, 3))
            if i == 0:
                data = [random_matrix]
            else:
                data = np.concatenate((data, [random_matrix]))

        self.stack = data

    # concatena le immagini in un insieme di matrici di punti
    # tutte le immagini in una matrice 3D
    def open_image(self):
        mat_a = None
        file_list = glob.glob("img/*.tif")
        for i in range(0, len(file_list)):
            image = Image.open(file_list[i])

            # call numpy to convert image in array of points
            image_array = np.array(image)

            # merge the arrays into one huger matrix
            if i == 0:
                mat_a = [image_array]
            else:
                mat_a = np.concatenate((mat_a, [image_array]))

        try:
            if mat_a.any():
                self.stack = mat_a
            else:
                raise IndexError
        except AttributeError as e:
            print(e)
            mat_a = np.array(mat_a)
            if mat_a.any():
                self.stack = mat_a
            else:
                raise IndexError


if __name__ == "__main__":
    id_fig = {}
    my_filter = ImageFilter()

    test_mode = 0
    if test_mode:
        my_filter.generate_test()
    else:
        my_filter.open_image()

    stack = my_filter.stack

    print(stack.shape)

    # extracting first image
    for i in range(len(stack)):

        data = stack[i, :, :, :]
        data = data.astype(np.int16)
        data_ori = data.copy()

        # crea figura e salva id delle stesse
        plt.figure(200)
        id_fig[10 + i] = [f"image_{10+i}"]

        # riconverte la matrice in immagine
        plt.imshow(data[:, :, :])
        plt.show()

        red = data.copy()
        red[:, :, 1] = 0.0
        red[:, :, 2] = 0.0
        # print(red.shape)

        green = data.copy()
        green[:, :, 0] = 0.0
        green[:, :, 2] = 0.0
        # print(green.shape)

        blue = data.copy()
        blue[:, :, 0] = 0.0
        blue[:, :, 1] = 0.0
        # print(blue.shape)
        print("")

        id_fig[10 + i].append({'reds': 1000 + i})
        # plt.figure(1000+i)
        # plt.imshow(red)
        # plt.show()
        
        id_fig[10 + i].append({'greens': 1100 + i})
        # plt.figure(1100+i)
        # plt.imshow(green)
        # plt.show()

        id_fig[10 + i].append({'blues': 1110 + i})
        # plt.figure(1110+i)
        # plt.imshow(blue)
        # plt.show()

        # rebuilt = reds + greens + blues
        plt.figure(10 + i)

        #rebuilt = rebuilt.astype(np.int16)

        reds = red[:, :, 0]
        greens = green[:, :, 1]
        blues = blue[:, :, 2]

        mask = ((red[:, :, 0] > my_filter.threshold) & (red[:, :, 0] > blue[:, :, 2]) & (reds > greens))
        mask2 = ((greens > my_filter.threshold) & (greens > reds) & (greens > blues))
        mask3 = ((blues > my_filter.threshold) & (blues > reds) & (greens < blues))

        n_mask = ((red[:, :, 0] < my_filter.threshold) & (red[:, :, 0] > blue[:, :, 2]) & (reds > greens))
        n_mask2 = ((greens < my_filter.threshold) & (greens > reds) & (greens > blues))
        n_mask3 = ((blues < my_filter.threshold) & (blues > reds) & (greens < blues))

        '''__________________'''
        print("original\n", data[:, :, :], "\n")
        '''elimina valori sotto soglia'''
        data[:, :, 0:3][n_mask] = 0.0
        data[:, :, 0:3][n_mask2] = 0.0
        data[:, :, 0:3][n_mask3] = 0.0

        '''modifica colori deboli quando color max supera soglia'''
        data[:, :, 1:][mask] -= my_filter.filter_force
        data[:, :, 0][mask2] -= my_filter.filter_force
        data[:, :, 2][mask2] -= my_filter.filter_force
        data[:, :, :2][mask3] -= my_filter.filter_force

        '''pavimento minimo valori = 0'''
        data[:, :, :][data[:, :, :] < 0] = 0.0

        data_high = data.copy()
        data_high = data_high.astype(np.uint8)

        '''reset the main matrix for further operations'''
        data = data_ori.copy()

        print(f"modified matrix according to max high color\n{data_high[:, :, :]}")

        print("")
        print(data[:, :, 0], data[:, :, 0].shape)
        print(data[:, :, 1], data[:, :, 1].shape)
        print(data[:, :, 2], data[:, :, 2].shape)

        print("")
        print("reds")
        print(data[:, :, 1:][n_mask].shape) #, data.shape)

        try:
            data[:, :, 1:][n_mask] = np.apply_along_axis(compare_rows, axis=0, arr=(data[:, :, 0][n_mask]), data=data[:, :, 1:][n_mask])
        except ValueError as e:
            print(f"errore rosso low: {e}")

        try:
            data[:, :, 0][n_mask2] = np.apply_along_axis(compare_rows, axis=0, arr=(data[:, :, 1][n_mask2]), data=data[:, :, 0][n_mask2])
        except ValueError as e:
            print(f"errore green_1 low: {e}")

        try:
            data[:, :, 2][n_mask2] = np.apply_along_axis(compare_rows, axis=0, arr=(data[:, :, 1][n_mask2]), data=data[:, :, 2][n_mask2])
        except ValueError as e:
            print(f"errore green_2 low: {e}")

        try:
            data[:, :, 1:][n_mask3] = np.apply_along_axis(compare_rows, axis=0, arr=(data[:, :, 2][n_mask3]), data=data[:, :, 0:2][n_mask3])
        except ValueError as e:
            print(f"errore blue low: {e}")

        data[:, :, 0:3][mask] = 0.0
        data[:, :, 0:3][mask2] = 0.0
        data[:, :, 0:3][mask3] = 0.0

        data_low = data.copy()
        data_low = data_low.astype(np.uint8)

        data_rebuilt = data_high + data_low

        plt.figure()
        plt.imshow(data_rebuilt)
        plt.show()

        print("\ndata low cleaned\n", data)




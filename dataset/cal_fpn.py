import numpy as np
from matplotlib import pyplot as plt


def cal_fpn_params(args):
    # load and process dark data
    L1 = args['dataset']['dark_lux']
    dark_img = np.load(args['dataset']['dark_data_path'])
    dark_img[dark_img == 0] = 1e-5
    D1 = 1 / dark_img
    # plt.imshow(dark_img, cmap='gray')
    # plt.axis('off')
    # plt.show()

    # load and process meduim data
    L2 = args['dataset']['medium_lux']
    medium_img = np.load(args['dataset']['medium_data_path'])
    D2 = 1 / medium_img
    # plt.imshow(meduim_img, cmap='gray')
    # plt.axis('off')
    # plt.show()

    # load and process light data
    L3 = args['dataset']['light_lux']
    light_img = np.load(args['dataset']['light_data_path'])
    D3 = 1 / light_img
    # plt.imshow(light_img, cmap='gray')
    # plt.axis('off')
    # plt.show()

    Ld = (L2 * D2 - L1 * D1) / (D1 - D2)  # Ld = (L2 * D2 - L1 * D1) / (D1 - D2)
    q = D1 * (L1 + Ld)

    # A = ((L3 + Ld(xm,ym)) * D3(xm,ym)) / ((L3 + Ld) * D3)
    A = ((L3 + Ld[args['dataset']['correct_pix_y'], args['dataset']['correct_pix_x']]) *
        D3[args['dataset']['correct_pix_y'], args['dataset']['correct_pix_x']]) / ((L3 + Ld) * D3)

    Ld_div_q = Ld / q

    return A, Ld_div_q

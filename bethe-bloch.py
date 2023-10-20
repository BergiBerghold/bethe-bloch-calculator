import numpy as np
from matplotlib.colors import LogNorm
from numpy import pi, log
import matplotlib.pyplot as plt

m_elec = 9.1093837e-31
q_elec = -1.60217663e-19
m_prot = 1.67262192e-27
c_0 = 299792458
N_A = 6.022e23
M_u = 1e-3
e_0 = 8.8541878128e-12

Al_6061 = {'Z': 13.0602, 'rho': 2700, 'A': 27.1069, 'I': 166.5276 * abs(q_elec)}
RaNO3 = {'Z': 59.52, 'rho': 4170 * 0.97, 'A': 150.64, 'I': 565.455 * abs(q_elec)}
Ra = {'Z': 88, 'rho': 5500, 'A': 225, 'I': 826 * abs(q_elec)}


def gaussian(x, A, sig):
    return A * np.exp( -x**2 / (2 * sig**2) )


def beta(E, m):
    v = np.sqrt(2*E / m)
    return v/c_0


def electron_density(Z, rho, A):
    return N_A * Z * rho / (A * M_u)


def calc_total_heat(radii, values):
    average = 0

    for r, v in zip(radii, values):
        circumference = 2 * r * np.pi
        average += v * circumference

    average /= len(radii)
    average /= max(radii) * np.pi

    print(f'{average:.1f} W/m^3')
    print(f'Total Heat: {average * 0.014**2 * pi * 0.002:.1f} W')


def bethe_bloch(E_0, stepsize, target):
    E = E_0
    total_energy = []
    energy_loss = []

    for item_0, item_1 in zip(target[:-1], target[1:]):
        x0 = item_0['x0']
        x1 = item_1['x0']

        material = item_0['material']
        I = material['I']
        n = electron_density(material['Z'], material['rho'], material['A'])

        for iteration in np.arange(x0, x1, stepsize):
            b = beta(E, m_prot)

            dE_dx = 4*pi / (m_elec * c_0**2)  *  n * 1 / b**2  *  (q_elec**2 / (4*pi*e_0))**2  *  ( log( 2*m_elec*c_0**2*b**2 / (I * (1-b**2)) ) - b**2 )

            E -= dE_dx * stepsize
            total_energy.append(E)
            energy_loss.append(dE_dx)

    return np.nan_to_num(total_energy), np.nan_to_num(energy_loss)


def generate_points(beam_sigma, beam_energy, beam_current, target_radius, target_stack, stepsize_r, stepsize_y):
    total_energy, energy_loss = bethe_bloch(E_0=beam_energy, stepsize=stepsize_y, target=target_stack)
    amplitude_gaussian = beam_current / (2*pi*beam_sigma**2)

    output_x = []
    output_y = []
    output_value = []

    graphics_array = []

    for r in np.arange(0, target_radius, stepsize_r):
        proton_flux = gaussian(x=r, A=amplitude_gaussian, sig=beam_sigma) / abs(q_elec)
        value = energy_loss * proton_flux

        if stepsize_r > stepsize_y:
            downsample = int(stepsize_r / stepsize_y)
            value = value[::downsample]

        output_x += list(np.linspace(0, target_stack[-1]['x0'], len(value)))
        output_y += [r] * len(value)
        output_value += list(value)

        graphics_array.append(value)

    with open('fluent_profile.prof', 'w') as f:
        f.write(f'((target point {len(output_x)})\n')
        f.write(f'(x {" ".join([str(x) for x in output_x])})\n')
        f.write(f'(y {" ".join([str(y) for y in output_y])})\n')
        f.write(f'(user-volumetric-energy-source {" ".join([str(v) for v in output_value])}))')

    return np.array(output_x), np.array(output_y), np.array(output_value), np.array(graphics_array)


# Generate ANSYS Pointsurface

if __name__ == '__main__':
    target_stack = [{'x0': 0, 'material': RaNO3}, {'x0': 5e-4, 'material': Al_6061}, {'x0': 2e-3, 'material': None}]

    output_x, output_y, output_value, graphics_array = generate_points(beam_sigma=2.7e-3,
                                                                       beam_energy=13e6 * abs(q_elec),
                                                                       beam_current=1e-6,
                                                                       target_radius=14e-3,
                                                                       target_stack=target_stack,
                                                                       stepsize_r=1e-5,
                                                                       stepsize_y=1e-8)


# Calculate Foil losses

# if __name__ == '__main__':
#     target_stack = [{'x0': 0, 'material': Al_6061}, {'x0': 25e-6, 'material': None}]
#
#     total_energy, energy_loss = bethe_bloch(E_0=13e6 * abs(q_elec), stepsize=1e-10, target=target_stack)
#
#     print(1 - min(total_energy) / max(total_energy))
#
#     print(min(total_energy) / abs(q_elec))


# Plotting

if __name__ == '__main__':
    # 2D Plot

    plt.imshow(np.concatenate((graphics_array.T[:,::-1], graphics_array.T), axis=1), extent=[-14,14,2,0], cmap='Oranges')
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 15
    cbar.set_label('Volumetric Heat Deposition [W / m3]', rotation=270)
    plt.xlabel('Target Y Coordinate [mm]')
    plt.ylabel('Target X Coordinate [mm]')
    plt.show()

    # Total Energy

    # print(calc_total_heat(output_y, output_value))
    #
    # Line plot
    #
    # step = 1e-8
    # target_stack = [{'x0': 0, 'material': RaNO3}, {'x0': 5e-4, 'material': Al_6061}, {'x0': 2e-3, 'material': None}]
    #
    # total_energy, energy_loss = bethe_bloch(13e6 * abs(q_elec), step, target_stack)
    #
    # fig, ax1 = plt.subplots()
    #
    # ax2 = ax1.twinx()
    # ax1.plot(np.arange(len(total_energy)) * step * 1000, total_energy / abs(q_elec) / 1e6, 'blue')
    # ax2.plot(np.arange(len(energy_loss)) * step * 1000, energy_loss / abs(q_elec), 'red')
    #
    # ax1.set_xlabel('\n\n\n\nDistance into Target')
    # ax1.set_ylabel('Total Proton Energy [MeV]', color='blue')
    # ax2.set_ylabel('Energy Loss per Proton per Distance [eV / m]', color='red')
    #
    # fig.set_size_inches(13, 7)
    # fig.tight_layout()
    # plt.show()


# Compare Plot

# if __name__ == '__main__':
#     step = 1e-8
#
#     target_stack_1 = [{'x0': 0, 'material': Al_6061}, {'x0': 2e-3, 'material': None}]
#     _, energy_loss_1 = bethe_bloch(13e6 * abs(q_elec), step, target_stack_1)
#
#     target_stack_2 = [{'x0': 0, 'material': RaNO3}, {'x0': 5e-4, 'material': Al_6061}, {'x0': 2e-3, 'material': None}]
#     _, energy_loss_2 = bethe_bloch(13e6 * abs(q_elec), step, target_stack_2)
#
#     target_stack_3 = [{'x0': 0, 'material': Ra}, {'x0': 5e-4, 'material': Al_6061}, {'x0': 2e-3, 'material': None}]
#     _, energy_loss_3 = bethe_bloch(13e6 * abs(q_elec), step, target_stack_3)
#
#     fig, ax = plt.subplots()
#
#     ax.plot(np.arange(len(energy_loss_1)) * step * 1000, energy_loss_1 / abs(q_elec), label='Pure Al')
#     ax.plot(np.arange(len(energy_loss_2)) * step * 1000, energy_loss_2 / abs(q_elec), label='0.5mm RaNO3')
#     ax.plot(np.arange(len(energy_loss_3)) * step * 1000, energy_loss_3 / abs(q_elec), label='0.5mm Ra Metal')
#
#     ax.set_xlabel('\n\n\n\nDistance into Target')
#     ax.set_ylabel('Energy Loss per Proton per Distance [eV / m]')
#     ax.legend(loc='best')
#
#     ax.set_xlim(xmin=-0.01, xmax=1.1)
#
#     fig.set_size_inches(13, 7)
#     fig.tight_layout()
#     plt.show()



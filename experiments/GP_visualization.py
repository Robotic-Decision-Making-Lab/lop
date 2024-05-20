# GP_visualization.py
# Written Ian Rankin December 2021
#
# A set of functions to visualize the learned GPs.

import numpy as np
import matplotlib.pyplot as plt
import pdb

import lop

def record_gp_state(model, fake_f, bounds=[(0,0),(1.5,1.5)], folder='./', \
                    file_header='', visualize=False):
    # Generate test grid
    num_side = 25
    x = np.linspace(bounds[0][0], bounds[0][1], num_side)
    y = np.linspace(bounds[1][0], bounds[1][1], num_side)

    X, Y = np.meshgrid(x,y)
    pts = np.vstack([X.ravel(), Y.ravel()]).transpose()


    fake_ut = fake_f(pts)
    max_fake = np.linalg.norm(fake_ut, ord=np.inf)
    fake_ut = fake_ut / max_fake

    pred_ut, pred_sigma = model.predict_large(pts)


    # save useful data
    # save entire pickle file
    gp_filename = folder+file_header+'_gp.p'
    print(gp_filename)
    #pickle.dump(gp, open(gp_filename, "wb"))
    # Save useful visualization data
    viz_filename = folder+file_header+'_viz'
    print(viz_filename)
    np.savez(viz_filename, \
            pts=pts, \
            fake_ut=fake_ut, \
            pred_ut=pred_ut, \
            pred_sigma=pred_sigma, \
            GP_pts=model.X_train, \
            GP_pref_0=model.y_train[0], \
            GP_pref_1=model.y_train[1], \
            GP_pref_2=model.y_train[2], \
            GP_prior_idx=model.prior_idx)


    if visualize == True:
        visualize_data(X, Y, num_side, fake_ut, pred_ut, pred_sigma,
            model.X_train, model.prior_idx, \
            folder, file_header, \
            also_display=False)


def visualize_data(X,Y, num_side, fake_ut, pred_ut, pred_sigma, \
                    GP_pts, GP_prior_idx, \
                    folder='./', file_header='', also_display=False):

    Z_pred = np.reshape(pred_ut, (num_side, num_side))
    Z_fake = np.reshape(fake_ut, (num_side, num_side))

    if pred_sigma is not None:
        Z_sigma = np.reshape(pred_sigma, (num_side, num_side))
        Z_std = np.sqrt(Z_sigma)
        Z_ucb = Z_pred + Z_std

    plt.figure()

    plt.pcolor(X, Y, Z_pred)
    plt.contour(X, Y, Z_pred)
    if GP_prior_idx is None:
        GP_prior_idx = [0,0]
    if GP_pts is not None:
        plt.scatter(GP_pts[GP_prior_idx[1]:,0],GP_pts[GP_prior_idx[1]:,1])
    else:
        plt.scatter([],[])
    plt.title(file_header+'_active samples')
    plt.xlabel('Migratory fish reward')
    plt.ylabel('Sea floor fish reward')

    plt.savefig(folder+file_header+'_active_samples.jpg')

    plt.figure()

    plt.pcolor(X, Y, Z_pred)
    plt.contour(X, Y, Z_pred)
    if GP_pts is not None:
        plt.scatter(GP_pts[:,0],GP_pts[:,1])
    else:
        plt.scatter([],[])
    plt.title(file_header+'_with prior points')
    plt.xlabel('Migratory fish reward')
    plt.ylabel('Sea floor fish reward')
    plt.savefig(folder+file_header+'_with_prior.jpg')

    if also_display:
        plt.show()

# GP_visualization.py
# Written Ian Rankin December 2021
#
# A set of functions to visualize the learned GPs.

import numpy as np
import matplotlib.pyplot as plt
import pdb

import lop

def record_gp_state(model, fake_f, bounds=[(0,0),(2.0,2.0)], folder='./', \
                    file_header='', visualize=False):
    
    if fake_f.w.shape[0] == 2:
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
                model.X_train, model.prior_idx, model.y_train[0], model.y_train[1], model.y_train[2], \
                folder, file_header, \
                also_display=False)
            
    else:
        viz_filename = folder+file_header+'_viz'
        np.savez(viz_filename, \
                GP_pts=model.X_train, \
                GP_pref_0=model.y_train[0], \
                GP_pref_1=model.y_train[1], \
                GP_pref_2=model.y_train[2], \
                GP_prior_idx=model.prior_idx)


def visualize_data(X,Y, num_side, fake_ut, pred_ut, pred_sigma, \
                    GP_pts, GP_prior_idx, GP_pref_0, GP_pref_1, GP_pref_2, \
                    folder='./', file_header='', also_display=False):

    Z_pred = np.reshape(pred_ut, (num_side, num_side))
    Z_fake = np.reshape(fake_ut, (num_side, num_side))

    if pred_sigma is not None:
        Z_sigma = np.reshape(pred_sigma, (num_side, num_side))
        Z_std = np.sqrt(Z_sigma)
        Z_ucb = Z_pred + Z_std

    plt.figure()
    ax = plt.gca()

    c = ax.pcolor(X, Y, Z_pred)
    ax.contour(X, Y, Z_pred)
    plt.colorbar(c, ax=ax)
    if GP_prior_idx is None:
        GP_prior_idx = [0,0]
    if GP_pts is not None:
        ax.scatter(GP_pts[GP_prior_idx[1]:,0],GP_pts[GP_prior_idx[1]:,1])
    else:
        ax.scatter([],[])
    plt.title(file_header+'_active samples')
    plt.xlabel('Migratory fish reward')
    plt.ylabel('Sea floor fish reward')

    alpha = 0.3
    color='red'
    width = 1.0
    head_size=15

    # Draw arrows
    if GP_pref_0 is not None:
        for pair in GP_pref_0:
            if pair[0] == lop.get_dk(1,0):
                lg_idx = pair[1]
                sm_idx = pair[2]
            else:
                lg_idx = pair[2]
                sm_idx = pair[1]
            sm_pt = GP_pts[sm_idx]
            lg_pt = GP_pts[lg_idx]

            diff = lg_pt - sm_pt
            loc=0.5
            line = ax.plot([sm_pt[0], lg_pt[0]], [sm_pt[1], lg_pt[1]], \
                    color=color, alpha=alpha, linewidth=width)[0]
            line.axes.annotate('',
                    xytext=(sm_pt[0]+diff[0]*loc, sm_pt[1]+diff[1]*loc),
                    xy=(sm_pt[0]+diff[0]*(loc+0.001), sm_pt[1]+diff[1]*(loc+0.001)),
                    arrowprops=dict(arrowstyle='->', color=color, alpha=alpha, lw=width),
                    size=head_size
            )


    plt.savefig(folder+file_header+'_active_samples.jpg')

    # plt.figure()

    # plt.pcolor(X, Y, Z_pred)
    # plt.contour(X, Y, Z_pred)
    # if GP_pts is not None:
    #     plt.scatter(GP_pts[:,0],GP_pts[:,1])
    # else:
    #     plt.scatter([],[])
    # plt.title(file_header+'_with prior points')
    # plt.xlabel('Migratory fish reward')
    # plt.ylabel('Sea floor fish reward')
    # plt.savefig(folder+file_header+'_with_prior.jpg')

    if also_display:
        plt.show()


# param score_diff - numpy array [iterations, num_evals]
def visualize_single_run_regret(folder, score_diff):
    avg_regret = np.mean(score_diff, axis=1)
    std_regret = np.std(score_diff, axis=1)
    std_error_mean = std_regret / np.sqrt(score_diff.shape[1])

    plt.figure()
    x = np.arange(0, score_diff.shape[0],dtype=int)
    ax = plt.gca()

    sigma_to_plot=1.0

    ax.fill_between(x,
                    avg_regret-(sigma_to_plot*std_error_mean),
                    avg_regret+(sigma_to_plot*std_error_mean),
                    alpha=0.1,
                    label='_nolegend_')
    ax.plot(x, avg_regret)

    plt.xticks(x)
    plt.ylabel('avg regret')
    plt.title('Regret over single run')

    plt.savefig(folder+'single_run_regret.jpg')



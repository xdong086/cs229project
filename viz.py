from plyfile import PlyData, PlyElement
import mayavi.mlab as mlab 
import numpy as np

def heading2rotmat(heading_angle):
        pass
        rotmat = np.zeros((3,3))
        rotmat[2,2] = 1
        cosval = np.cos(heading_angle)
        sinval = np.sin(heading_angle)
        rotmat[0:2,0:2] = np.array([[cosval, -sinval],[sinval, cosval]])
        return rotmat

def get_3d_box(box_size, heading_angle, center):
    ''' Calculate 3D bounding box corners from its parameterization.

    Input:
        box_size: tuple of (l,w,h)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''
    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])

    #R = roty(heading_angle)
    R = heading2rotmat(heading_angle)
    print(box_size.shape)
    l,w,h = box_size
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + center[0];
    corners_3d[1,:] = corners_3d[1,:] + center[1];
    corners_3d[2,:] = corners_3d[2,:] + center[2];
    corners_3d = np.transpose(corners_3d)
    return corners_3d

def draw_gt_boxes3d(gt_boxes3d, fig, color=(1,1,1), line_width=1, draw_text=False, text_scale=(1,1,1), color_list=None):
    ''' Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    ''' 
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n] 
        if draw_text: mlab.text3d(b[4,0], b[4,1], b[4,2], '%d'%n, scale=text_scale, color=color, figure=fig)
        for k in range(0,4):
            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+1)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)
    #mlab.show(1)
    #mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig

def viz_2d():
	return

def viz_3d(pc_file, gt_bbox_file, pred_bbox_file):
	pc = np.load(pc_file)
	gt_bbox = np.load(gt_bbox_file)
	pred_bbox = np.load(pred_bbox_file)

	# fig_gt = mlab.figure(figure=None, bgcolor=(1.0,1.0,1.0), fgcolor=None, engine=None, size=(1000, 800))
	# mlab.points3d(pc[:,0], pc[:,1], pc[:,2], color=(0.1, 0.1, 0.1), mode='point', colormap='gnuplot', scale_factor=1, figure=fig_gt)
	
	# for idx in range(gt_bbox.shape[0]):
	# 	box3d_from_label_gt = get_3d_box(gt_bbox[idx, 3:6], gt_bbox[idx, 6],gt_bbox[idx,0:3])
	# 	draw_gt_boxes3d([box3d_from_label_gt], fig_gt, color=(0,0,1))

	fig_pred = mlab.figure(figure=None, bgcolor=(1.0,1.0,1.0), fgcolor=None, engine=None, size=(1000, 800))
	mlab.points3d(pc[:,0], pc[:,1], pc[:,2], color=(0.1, 0.1, 0.1), mode='point', colormap='gnuplot', scale_factor=1, figure=fig_pred)

	for idx in range(pred_bbox.shape[0]):
		box3d_from_label_pred = get_3d_box(pred_bbox[idx, 3:6], pred_bbox[idx, 6], pred_bbox[idx,0:3])
		draw_gt_boxes3d([box3d_from_label_pred], fig_pred, color=(0,0,1))

	input()
	return

if __name__=='__main__':
	viz_3d("/samsung/votenet/eval_sunrgbd/000004_pc.npy",
		"/samsung/votenet/eval_sunrgbd/000004_gt_bbox_obbs.npy",
		"/samsung/votenet/eval_sunrgbd/000004_pred_bbox_obbs.npy")
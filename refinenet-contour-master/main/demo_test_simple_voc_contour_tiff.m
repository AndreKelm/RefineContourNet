
% Author: Guosheng Lin (guosheng.lin@gmail.com)
% Modified by: André Kelm and Vijesh Rao

% this is a simpler demo file for testing on your own images.

function demo_test_simple_voc_contour()

rng('shuffle');
addpath('./my_utils');
dir_matConvNet='../libs/matconvnet/matlab';
% run(fullfile(dir_matConvNet, 'vl_setupnn.m'));

run_config=[];

run_config.use_gpu=true;
% run_config.use_gpu=false;
run_config.gpu_idx=1;

% result dir:
result_name=['runner_result_dir' datestr(now, 'YYYYmmDDHHMMSS')];
result_dir=fullfile('../cache_data', 'result_on_test_imgs', result_name);
mkdir_notexist(result_dir);

% the folder that contains testing images:
img_data_dir='../datasets/example_imgs_bsds500';

% using a trained model which is trained on Contour Detection
run_config.trained_model_path='../model_trained/RCN_edge_detection.mat';
%run_config.class_info=gen_class_info_voc();
run_config.class_info=gen_class_info_contour();

% for trained model, control the size of input images
run_config.input_img_short_edge_min=450;
run_config.input_img_short_edge_max=800;

runner_info=prepare_runner_test_simple(run_config);

img_filenames=my_list_file(img_data_dir);
img_num=length(img_filenames);
for img_idx=1:img_num
    task_info=[];
    task_info.img_dir=img_data_dir;
    task_info.img_filename=img_filenames{img_idx};
    task_result=runner_info.run_task_fn(runner_info, task_info);
    
    [~, img_name]=fileparts(task_info.img_filename);
    one_cache_file=fullfile(result_dir, [img_name '.png']);
    fprintf('save predict mask:%s\n', one_cache_file);
    % imwrite(task_result.mask_data, run_config.class_info.mask_cmap, one_cache_file);
    
    score_map_HD = task_result.score_map_HD ; 
    img_size = task_result.img_size ;
    score_map_HD_resized=my_resize(score_map_HD, img_size);
        
    soft_map = imresize(score_map_HD,img_size);
    soft_map =  1./(1 + exp(-soft_map)) ;
    soft_map = im2uint16(soft_map);
       
    Sc_map_path= fullfile(result_dir, [img_name '.tiff']) ;
    
    imwrite(soft_map, Sc_map_path, 'tiff' ) ;

end

end



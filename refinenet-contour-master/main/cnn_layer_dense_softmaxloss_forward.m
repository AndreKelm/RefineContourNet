


function output_info=cnn_layer_dense_softmaxloss_forward(input_info, layer, work_info_batch)

    
    stageout_info=layer.stageout_info;
    stage_idx=stageout_info.stage_idx;
       
    mc_info=gen_mc_info(input_info, work_info_batch);
    work_info_batch.ref.finalloss_mc_info_stages{stage_idx}=mc_info;
    example_non_valid_flags=mc_info.example_non_valid_flags;
    
    tmp_input_info=input_info;
    work_info_batch.ref.finalloss_tmp_input_info_stages{stage_idx}=tmp_input_info;
    
    output_info=do_forward(tmp_input_info, mc_info);
            
    if ~isempty(example_non_valid_flags)
        valid_r_num=numel(example_non_valid_flags)-nnz(example_non_valid_flags);
        if valid_r_num>0
            output_info.x=output_info.x./valid_r_num;
        end
                
        valid_node_ratio=valid_r_num./numel(example_non_valid_flags);
    else
        valid_node_ratio=1;
        %valid_r_num=mc_info.node_num;
        output_info.x=output_info.x;
    end
    
    
    if stageout_info.gen_prediction_info
    %tmp_input_info.x contains the output of the network
        output_info.mc_predict_info=gen_predict_info(mc_info, tmp_input_info.x);
    end
    
            
    task_run_info=[];
    task_run_info.task_finished=true;
    task_run_info.task_finish_progress=1;
    task_run_info.todo_node_flags=[];
    make_ref_obj(task_run_info);
    work_info_batch.ref.task_run_info=task_run_info;
    
    work_info_batch.ref.node_num=mc_info.node_num;
    work_info_batch.ref.valid_node_ratio=valid_node_ratio;
    work_info_batch.ref.predict_map_size=mc_info.node_map_size;
        
        
end



function one_gt_mask_coarse=do_gen_gt_mask_data(work_info_batch, node_map_size)

%if work_info_batch.ref.train_opts.init_resnet_layer_num == 50  
%    one_gt_mask_coarse = work_info_batch.ref.ds_info.batch_data.label_data == 1;
%    return;
%end
batch_ds_info=work_info_batch.ref.ds_info;
one_gt_mask=batch_ds_info.batch_data.label_data;
se = strel('disk',1);
dilated_img = imdilate(one_gt_mask, se);
one_gt_mask_coarse=imresize(dilated_img, node_map_size, 'bicubic');
one_gt_mask_coarse = one_gt_mask_coarse ==1;
%figure, imshow(one_gt_mask_coarse == 1);
end



function mc_info=gen_mc_info(input_info, work_info_batch)

input_feat_map_size=size(input_info.x);
node_map_size=input_feat_map_size(1:2);
node_label_data=do_gen_gt_mask_data(work_info_batch, node_map_size);

class_info=work_info_batch.ref.imdb.ref.ds_info.class_info;

exclude_node_flags=gen_exclude_node_flags(class_info, node_label_data);
work_info_batch.ref.exclude_node_flags=exclude_node_flags;
if ~isempty(exclude_node_flags)
    % void label supported by matconvnet
    invalid_class_label_value=0;
    node_label_data(exclude_node_flags)=invalid_class_label_value;
end

class_num=class_info.class_num;
%assert(class_num==input_feat_map_size(3));

mc_info=[];
mc_info.class_num=class_num;
mc_info.node_map_size=node_map_size;
mc_info.node_num=prod(node_map_size);
mc_info.e_num=mc_info.node_num;

mc_info.gt_label_data=double(node_label_data);
mc_info.example_non_valid_flags=exclude_node_flags;

if ~isempty(exclude_node_flags)
    mc_info.valid_e_num=numel(exclude_node_flags)-nnz(exclude_node_flags);
else
    mc_info.valid_e_num=mc_info.node_num;
end



end



function predict_info=gen_predict_info(mc_info, predict_scores)
            
    my_check_valid_numeric(predict_scores);
    map_size=size(predict_scores);
    
    %assert(map_size(3)==mc_info.class_num);
    assert(all(map_size(1:2)==mc_info.node_map_size));
    
    predict_scores=single(predict_scores);
    
    %score predicted is the op of the sigmoid
    predict_scores=vl_nnsigmoid(predict_scores, []);
        
    predict_info=[];
    predict_info.mc_info=mc_info;
    predict_info.score_map=predict_scores;
        
end





function exclude_node_flags=gen_exclude_node_flags(class_info, node_label_data)


exclude_class_idxes=class_info.void_class_idxes;

exclude_node_flags=[];

if ~isempty(exclude_class_idxes)
      
   for ex_idx=1:length(exclude_class_idxes)
        one_exclude_flags=node_label_data==exclude_class_idxes(ex_idx);
                
        if my_any(one_exclude_flags)
            if isempty(exclude_node_flags)
                exclude_node_flags=one_exclude_flags;
            else
                exclude_node_flags=exclude_node_flags|one_exclude_flags;
            end
        end
   end
end


end



function output_info=do_forward(input_info, mc_info)

assert(isa(mc_info.gt_label_data, 'double'));

%sigmoid_op = vl_nnsigmoid(input_info.x, []);

mask = mc_info.gt_label_data == 0;
mc_info.gt_label_data ( mask ) = -1;
instanceWeights = double(mc_info.gt_label_data == 1)*10 + double(mc_info.gt_label_data == -1);
output_x = vl_nnloss(input_info.x, mc_info.gt_label_data, [], 'loss', 'logistic', 'instanceWeights',instanceWeights ) ;

output_info=[];
output_info.is_group_data=false;
output_info.x=output_x;
output_info.sigmoid = input_info.x;

assert(size(input_info.x,3)==mc_info.class_num);


end




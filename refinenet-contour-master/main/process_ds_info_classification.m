

function ds_info=process_ds_info_classification(ds_info, ds_config)
class_info=ds_info.class_info;

img_num=length(ds_info.img_names);
class_idxes_mask_dir=ds_info.data_dirs{2};


class_label_values=class_info.class_label_values;
assert(isa(class_label_values, 'uint8'));

class_num=length(class_label_values);
assert(class_num<2^8);


mask_files=cell(img_num, 1);
pixel_count_classes=zeros(class_num, 1);

assert(~ds_config.use_dummy_gt)
class_idxes_imgs = cell(img_num,1);

for img_idx=1:img_num
    
    mask_data=load_mask_from_ds_info(ds_info, img_idx);
 
    pixel_count_classes(img_num)=nnz(mask_data);


    mask_file_name=ds_info.img_files{img_idx};
    [~, mask_file_name]=fileparts(mask_file_name);
    mask_files{img_idx}=[mask_file_name '.png']; 
    class_idxes_imgs{img_idx} = 1;
end



class_idxes_mask_data_info=[];
class_idxes_mask_data_info.mask_files=mask_files;
class_idxes_mask_data_info.data_dirs={class_idxes_mask_dir}';
class_idxes_mask_data_info.data_dir_idxes_mask=ones(img_num, 1, 'uint8');


ds_info.class_idxes_mask_data_info=class_idxes_mask_data_info;
ds_info.class_idxes_imgs=class_idxes_imgs;

% ds_info.class_label_values_imgs=class_label_values_imgs;
ds_info.pixel_count_classes=pixel_count_classes';

end






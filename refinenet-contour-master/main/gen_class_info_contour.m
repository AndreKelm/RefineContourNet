
function class_info=gen_class_info_contour()

class_info=[];

class_info.class_names = {'boat'}; 
        

%void_class_value=255;
class_info.class_label_values=uint8(1);
%class_info.background_label_value=uint8(0);
%class_info.void_label_values=uint8(void_class_value);

% addpath ../libs/VOCdevkit_2012/VOCcode
class_info.mask_cmap = VOClabelcolormap(256);

class_info=process_class_info(class_info);

end
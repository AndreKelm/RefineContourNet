function gen_contours_boat()

boat_counter = 0;
boat_list = [];

contour_dir = 'D:\TUHH\Arbeit_contours\benchmark_RELEASE\dataset\cls\';
path_contour = 'D:\TUHH\Arbeit_contours\benchmark_RELEASE\boat_contour\';
path_segmentation = 'D:\TUHH\Arbeit_contours\benchmark_RELEASE\boat_segementation\';
list = dir(contour_dir);
list = list([3:end]);
cell_my = struct2cell(list);
num_elements = numel(list);
for i = 1:num_elements
    filename = cell_my(1,i);
    filename = filename{1};
    progressbar(i/num_elements);
    load(strcat(contour_dir,filename));
    if GTcls.CategoriesPresent == 4
        boat_counter = boat_counter + 1;
        filename_png = strcat(regexprep(filename,'[.mat]',''), '.png');
        imwrite(GTcls.Segmentation,VOClabelcolormap,strcat(path_segmentation,filename_png));
        imwrite(full(GTcls.Boundaries{4,1}),strcat(path_contour,filename_png));
    end
end
disp(boat_counter);

end
    
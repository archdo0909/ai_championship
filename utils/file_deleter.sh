target_path="/workspace/eddie/ai_championship/data/lg_train/"
for entry in "$target_path"/* 
do
    t_dir="$target_path${entry##*/}"
    #echo "$t_dir"
    for folder in "$t_dir"/*
    do
        folder_name="${folder##*/}"
        #echo "${folder_name%%.*}"
        
        t_file="$t_dir/${folder_name##*/}"
        echo "$t_file"
        rm "$t_file"/*_split_*
    done
    #echo "Next"
done
# echo "Done"

#!/bin/bash -e
target_path="/workspace/eddie/ai_championship/data/202003/"

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
        for file in "$t_file"/*
        do
            case "$file" in (*_modi_train_*)
                echo "${file%%.*}"
                #rm "$file"
                #split --additional-suffix=.txt -a 4 -l 5000 -d "$file" "${file%%.*}_split_"
            esac
        done
    done
    #echo "Next"
done
echo "Done"

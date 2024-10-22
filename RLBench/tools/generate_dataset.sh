splits=(
'val'
)

for split in "${splits[@]}"
do amount=25
save_path="/scratch/one_month/current/davidyyd/data/$split"
tasks=(
#open_drawer
#slide_block_to_color_target
#sweep_to_dustpan_of_size
#meat_off_grill
#turn_tap
#put_item_in_drawer
close_jar
#reach_and_drag
#stack_blocks
#light_bulb_in
#put_money_in_safe
#place_wine_at_rack_location
#put_groceries_in_cupboard
#place_shape_in_shape_sorter
#push_buttons
#insert_onto_square_peg
#stack_cups
#place_cups
)

for task in "${tasks[@]}"
do DISPLAY=:0.0 python dataset_generator.py --tasks=$task \
                            --save_path=$save_path \
                            --renderer=opengl \
                            --episodes_per_task=$amount \
                            --processes=1 \
                            --variations=1 \
                            --all_variations=False
done
done

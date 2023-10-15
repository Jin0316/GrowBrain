TARGET = 'test'

with open(f'/workspace/video_dataset/kinetics/k400_rgb/missing_info/{TARGET}_missing_vids_list.txt', 'r') as f:
    lines = f.readlines()

with open(f'/workspace/video_dataset/kinetics/k400_rgb/repl_info/{TARGET}_non_repl_vids.txt', 'r') as f:
    lines2 = f.readlines()

print(len(lines))
print(len(lines2))
#print(lines2)


with open(f'./missing_{TARGET}.txt', 'w') as f:
    for l in lines:
        f.write(l)

    for l in lines2:
        f.write(l.split('/')[-1])

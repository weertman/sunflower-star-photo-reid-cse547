import os
import glob

model_root = os.path.join('..','..','models','reid')
print(os.path.exists(model_root))

model_settings = glob.glob(os.path.join(model_root, '*', 'logs', 'settings.txt'))
print(f'Found {len(model_settings)} settings files')

for setting_file in model_settings[:1]:

    model_name = os.path.basename(os.path.dirname(os.path.dirname(setting_file)))
    print(f'Model directory: {model_name}')
    dataset = model_name.split('__')[0]
    print(f'Dataset: {dataset}')

    ## open settings file
    ## check if Dataset is in the file ## f.write(f'Dataset: {name_data_dir}\n') # is how it is written
    ## if not, add it to the file by inserting it after the second line
    added = False
    with open(setting_file, 'r') as file:
        lines = file.readlines()
        if 'Dataset' not in lines[1]:
            lines.insert(1, f'Dataset: {dataset}\n')
            added = True
        else:
            print(f'Dataset already in settings file: {setting_file}')

    if added == True:
        with open(setting_file, 'w') as file:
            file.writelines(lines)
        print(f'Updated settings file: {setting_file}')
    else:
        print(f'No changes made to settings file: {setting_file}')



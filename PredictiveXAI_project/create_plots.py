
import re
import matplotlib.pyplot as plt
import os

# --- Main Configuration ---
# CHANGE THESE VALUES TO SWITCH BETWEEN MODELS
PROTOTYPE_COUNT = 200  # Options: 200 or 2000
BASE_LOG_DIR = 'saved_models/resnet34' # Base directory for your logs

# --- Construct file paths and names ---
MODEL_NAME = f'{PROTOTYPE_COUNT} prototypes'
LOG_FILE_NAME = os.path.join(BASE_LOG_DIR, f'protopnet_{PROTOTYPE_COUNT}proto', 'train.log')


def parse_log_file(filename):
    """
    Parses the ProtoPNet log file, creating a stretched x-axis for clarity
    during push and iteration phases.
    """
    train_points = []
    test_points = []
    tick_map = {}  # Maps original epoch number to its new x-coordinate
    shade_regions = [] # Stores start/end coordinates for shaded backgrounds

    try:
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        print("Please check the PROTOTYPE_COUNT and BASE_LOG_DIR variables.")
        return None, None, None, None

    # --- State variables for parsing ---
    current_epoch = -1
    is_in_iteration_phase = False
    x_step = -1 # Start at -1 so the first epoch (0) begins at x_step 0
    
    # --- Pass 1: Map epochs to x-coordinates to handle the stretch ---
    temp_epoch = -1
    for line in lines:
        epoch_match = re.search(r'^epoch:\s*(\d+)', line)
        if epoch_match:
            new_epoch = int(epoch_match.group(1))
            if new_epoch != temp_epoch:
                temp_epoch = new_epoch
                x_step += 1
                tick_map[temp_epoch] = x_step
                if temp_epoch > 0 and temp_epoch % 50 == 0:
                    shade_regions.append((x_step, x_step + 20))
                    x_step += 20 # Add the stretch
    
    # --- Pass 2: Extract data using the coordinate map ---
    for i, line in enumerate(lines):
        epoch_match = re.search(r'^epoch:\s*(\d+)', line)
        iteration_match = re.search(r'^iteration:\s*(\d+)', line)

        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            is_in_iteration_phase = False
        elif iteration_match:
            is_in_iteration_phase = True
            current_iteration = int(iteration_match.group(1))

        if line in ['train', 'test']:
            accu = None
            for j in range(i + 1, min(i + 10, len(lines))):
                accu_match = re.search(r'accu:\s*([\d.]+)%', lines[j])
                if accu_match:
                    accu = float(accu_match.group(1))
                    break
            if accu is None:
                continue

            if is_in_iteration_phase:
                plot_x = tick_map[current_epoch] + current_iteration + 1
            else:
                plot_x = tick_map[current_epoch]

            if line == 'train':
                category = 'Iteration Training' if is_in_iteration_phase else 'Epoch Training'
                train_points.append((plot_x, accu, category))
            elif line == 'test':
                if is_in_iteration_phase:
                    category = 'Iteration Testing'
                elif 'Executing push ...' in lines[i+2] or 'Executing push ...' in lines[i+3]:
                    category = 'Pre-Push Test'
                elif 'push time:' in lines[i-1] or 'push time:' in lines[i-2]:
                    category = 'Post-Push Test'
                    plot_x += 0.5
                else:
                    category = 'Epoch Testing'
                test_points.append((plot_x, accu, category))

    return train_points, test_points, tick_map, shade_regions


def create_plots(train_data, test_data, tick_map, shade_regions):
    """
    Generates and saves plots with a stretched axis and shaded regions.
    """
    if not train_data and not test_data:
        print("No data was parsed. Cannot create plots.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')

    # --- Generate Tick Labels for the custom x-axis ---
    tick_locs, tick_labels = [], []
    max_epoch = max(tick_map.keys()) if tick_map else 0
    
    # Define which epochs to add to the x-axis
    tick_epochs = set([0] + list(range(10, max_epoch + 1, 10)))
    # Add push epochs and the epoch immediately after
    for epoch in sorted(tick_map.keys()):
        if epoch > 0 and epoch % 50 == 0:
            tick_epochs.add(epoch)
            if epoch + 1 in tick_map:
                tick_epochs.add(epoch + 1)
    
    # Create the lists for plotting
    for epoch in sorted(list(tick_epochs)):
        if epoch in tick_map:
            tick_locs.append(tick_map[epoch])
            tick_labels.append(str(epoch))

    # --- 1. Plot Training Accuracy ---
    fig_train, ax_train = plt.subplots(figsize=(16, 9))
    epoch_train = [p for p in train_data if p[2] == 'Epoch Training']
    iter_train = [p for p in train_data if p[2] == 'Iteration Training']

    if epoch_train:
        ax_train.plot([p[0] for p in epoch_train], [p[1] for p in epoch_train],
                  marker='o', linestyle='-', label='Epoch Training', color='royalblue')
    if iter_train:
        ax_train.plot([p[0] for p in iter_train], [p[1] for p in iter_train],
                  marker='x', linestyle=':', label='Iteration Training (Last Layer)', color='darkorange')

    for start, end in shade_regions:
        ax_train.axvspan(start, end, color='grey', alpha=0.1, zorder=0, label='_nolegend_')

    ax_train.set_title(f'Training Accuracy Over Time ({MODEL_NAME})', fontsize=18, pad=20)
    ax_train.set_xlabel('Epoch Number', fontsize=12)
    ax_train.set_ylabel('Accuracy (%)', fontsize=12)
    ax_train.legend(fontsize=11)
    ax_train.set_xticks(tick_locs, tick_labels, rotation=45)
    ax_train.grid(True, which='major', axis='x', linestyle='-') # Emphasize major ticks
    fig_train.tight_layout()
    fig_train.savefig(f'train_accuracy_plot_{PROTOTYPE_COUNT}.png', dpi=300)
    print(f"Saved training accuracy plot to 'train_accuracy_plot_{PROTOTYPE_COUNT}.png'")
    plt.close(fig_train)

    # --- 2. Plot Test Accuracy ---
    fig_test, ax_test = plt.subplots(figsize=(16, 9))
    epoch_test = [p for p in test_data if p[2] == 'Epoch Testing']
    iter_test = [p for p in test_data if p[2] == 'Iteration Testing']
    pre_push = [p for p in test_data if p[2] == 'Pre-Push Test']
    post_push = [p for p in test_data if p[2] == 'Post-Push Test']
    
    if epoch_test:
        ax_test.plot([p[0] for p in epoch_test], [p[1] for p in epoch_test],
                 marker='o', linestyle='-', label='Epoch Testing', color='seagreen', zorder=2)
    if iter_test:
        ax_test.plot([p[0] for p in iter_test], [p[1] for p in iter_test],
                 marker='x', linestyle=':', label='Iteration Testing (Last Layer)', color='darkorange', zorder=2)
    if pre_push:
        ax_test.scatter([p[0] for p in pre_push], [p[1] for p in pre_push],
                    marker='v', label='Pre-Push Test', color='crimson', s=120, zorder=4, edgecolors='black')
    if post_push:
        ax_test.scatter([p[0] for p in post_push], [p[1] for p in post_push],
                    marker='^', label='Post-Push Test', color='mediumpurple', s=120, zorder=4, edgecolors='black')

    for start, end in shade_regions:
        ax_test.axvspan(start, end, color='grey', alpha=0.1, zorder=0, label='_nolegend_')

    ax_test.set_title(f'Test Accuracy Over Time ({MODEL_NAME})', fontsize=18, pad=20)
    ax_test.set_xlabel('Epoch Number', fontsize=12)
    ax_test.set_ylabel('Accuracy (%)', fontsize=12)
    ax_test.legend(fontsize=11)
    ax_test.set_xticks(tick_locs, tick_labels, rotation=45)
    ax_test.grid(True, which='major', axis='x', linestyle='-') # Emphasize major ticks
    fig_test.tight_layout()
    fig_test.savefig(f'test_accuracy_plot_{PROTOTYPE_COUNT}.png', dpi=300)
    print(f"Saved test accuracy plot to 'test_accuracy_plot_{PROTOTYPE_COUNT}.png'")
    plt.close(fig_test)


if __name__ == '__main__':
    train_data, test_data, tick_map, shade_regions = parse_log_file(LOG_FILE_NAME)
    if train_data is not None:
        create_plots(train_data, test_data, tick_map, shade_regions)
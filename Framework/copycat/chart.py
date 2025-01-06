import os
import subprocess
import csv
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# Ścieżki do skryptów i plików
TARGET_MODEL = "../model_cifar_resnet.pth"
COPYCAT_MODEL = "copycat.pth"
IMAGE_LIST = "images.txt"
LABELS = "stolen_labels.txt"
CSV_OUTPUT = "results.csv"

# Klasa modelu zgodna z ResNet18
class ResNetModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetModel, self).__init__()

        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


def run_command_output(command):
    result = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, text=True)
    result.wait()  # Wait for the process to complete
    if result.returncode != 0:
        print(f"Error running command: {command}", file=sys.stderr)
    output = result.stdout
    return output

def run_command(command):
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"Error running command: {command}", file=sys.stderr)


# Główna logika skryptu
def main():
    # Wczytanie modelu docelowego, aby upewnić się, że wagi są zgodne
    model = ResNetModel(num_classes=10)
    try:
        model.load_state_dict(torch.load(TARGET_MODEL, map_location=torch.device('cpu')))
    except RuntimeError as e:
        print(f"Error loading model weights: {e}")
        return

    with open(CSV_OUTPUT, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Number of Images", "Accuracy"])

        for num_images in range(7500, 40001, 500):
            print(f"Processing {num_images} images...")

            # # Generowanie etykiet dla podanej liczby obrazów
            command_label = (
                f"python3 -W ignore::FutureWarning copycat/label_data.py {TARGET_MODEL} {IMAGE_LIST} {LABELS} {num_images}"
            )
            run_command(command_label)

            # Trenowanie modelu Copycat
            command_train = f"python3 -W ignore::FutureWarning copycat/train.py {COPYCAT_MODEL} {LABELS}"
            run_command(command_train)

            # Testowanie modelu Copycat
            command_test = f"python3 -W ignore::FutureWarning oracle/test.py {COPYCAT_MODEL}"
            output = run_command_output(command_test)
            content = output.read()

            # Wyodrębnianie accuracy z wyniku testu
            accuracy_line = [line for line in content.split('\n') if "%" in line]
            line = accuracy_line[0]
            accuracy = float(line.split(":")[1].split("%")[0].strip())

            # Zapisywanie wyników do pliku CSV
            writer.writerow([num_images, accuracy])

if __name__ == "__main__":
    main()
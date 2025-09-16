# Process starts: Imports.
import os  # OS.
import datetime  # Time.
# Process ends.

# Process starts: Logger class.
class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir  # Dir.
        os.makedirs(log_dir, exist_ok=True)  # Make dir.
        self.log_file = os.path.join(log_dir, f'log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')  # File.

    def log(self, message):
        print(message)  # Print.
        with open(self.log_file, 'a') as f:  # Append.
            f.write(message + '\n')  # Write.
# Process ends.
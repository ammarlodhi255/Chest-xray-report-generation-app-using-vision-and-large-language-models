import subprocess
import os
import signal

def kill_port_linux(port):
    try:
        # Find processes using the port
        result = subprocess.check_output(f'lsof -i :{port}', shell=True).decode()
        lines = result.strip().split('\n')
        for line in lines[1:]:  # skip header line
            parts = line.split()
            pid = int(parts[1])
            os.kill(pid, signal.SIGKILL)
            print(f'Killed PID {pid} using port {port}')
    except subprocess.CalledProcessError:
        print(f'No process is using port {port}.')
    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    kill_port_linux(5000)
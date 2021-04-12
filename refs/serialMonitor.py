import serial
import time
import re

def Reader():
    global ser
    if (ser.isOpen()):
        try:                    
            x = ser.readline().decode()
            x = (x)
            return x
        except:
            return "Unable to print\n"
    else: 
        return "Cannot open serial port\n"
    
def SerialSetup():
    #Open serial port to read from Arduino serial monitor    
    arduino = serial.Serial('/dev/mcu_aux', 9600, timeout=.1) #COM9 = left hand side USB
    time.sleep(1) #give the connection a second to settle

def SerialDistances():
    data = arduino.readline()[:-2].decode() #[:~2] removes newline chars
    values = re.split(r'\t',data)
    leftDist = values[0]
    rightDist= values[1]
    return leftDist,rightDist

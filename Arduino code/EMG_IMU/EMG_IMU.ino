/*
Online GUID / UUID Generator:
https://www.guidgenerator.com/online-guid-generator.aspx
eg.64cf715d-f89e-4ec0-b5c5-d10ad9b53bf2
*/

#include <ArduinoBLE.h> // Arduino BLE library https://www.arduino.cc/reference/en/libraries/arduinoble/
#include "LSM6DS3.h"    // IMU Library 6DoF
#include "Wire.h"       // 

#define CONVERT_G_TO_MS2 9.80665f // g
#define FREQUENCY_HZ 50 // sample frequence
#define INTERVAL_MS (1000 / (FREQUENCY_HZ))
//---------------------------------------------
#define Serial_flag 0 //log标志位
#define device_id 1 //设备编号
//---------------------------------------------
#if device_id == 1
const char* UUID_serv = "1ec1cf63-9f15-4764-8d94-4ffbf801adb6"; // UUID for Service
const char* setLocalNameTOBE = "IMU-1-Project"; // setLocalName
#elif device_id == 2
const char* UUID_serv = "d2773a14-faee-4b3a-8581-5f08d584f504"; 
const char* setLocalNameTOBE = "IMU-2-Project";
#endif

// UUID for accelerometer characteristics
const char* UUID_acc_gyro = "af879017-8c9c-4092-8da1-0d115d08fa79";

// BLE Service
BLEService myService(UUID_serv); 

// BLE Characteristics
BLECharacteristic chAccGyro(UUID_acc_gyro,BLERead|BLENotify,50);

//Create an instance of class LSM6DS3
LSM6DS3 xIMU(I2C_MODE, 0x6A);    //I2C device address 0x6A

float dataArray[7] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0}; //初始化有用到
int EMG_data;
int a =0;
const int analogPin = A0; // 定义模拟输入引脚
int value = 0;            // 存储读取的值

void setup() 
{

  pinMode(LEDR, OUTPUT); // onboard led red
  pinMode(LEDG, OUTPUT); // onboard led green
  pinMode(LEDB, OUTPUT); // onboard led blue 
  digitalWrite(LEDR, HIGH); // led red off
  digitalWrite(LEDG, HIGH); // led green off
  digitalWrite(LEDB, HIGH); // led blue off

  showID(device_id); // show ID using led blue  

  if(Serial_flag){
    Serial.begin(115200);
    while (!Serial);
    // Serial.print("Seeed XIAO BLE Sense IMU-Acc-gyro Data Logger");
  }
  
  bool err=false;
  
  // init IMU
  if (xIMU.begin() != 0) {
    // Serial.print("Device error");
    err = true;
  } else {
    // Serial.print("Device OK!");
  }

  // init BLE
  if (!BLE.begin()) 
  {
    // Serial.print("BLE: failed");
    err=true;
  }
  // Serial.print("BLE: ok");

  // error: flash led forever
  if (err)
  {
    // Serial.print("Init error. System halted");
    while(1)
    {
      digitalWrite(LEDR, LOW);
      delay(500); 
      digitalWrite(LEDR, HIGH); // led on
      delay(500);
    } 
  }

  // Set BLE name
  BLE.setLocalName(setLocalNameTOBE);
  BLE.setDeviceName("XIAO-BLE-Sense"); 
  
  // Set advertised Service
  BLE.setAdvertisedService(myService);
  
  // Add characteristics to the Service
  myService.addCharacteristic(chAccGyro);
  
  // add service to BLE
  BLE.addService(myService);
  
  // characteristics initial values
  chAccGyro.writeValue(dataArray,7);
 
  // start advertising
  BLE.advertise();
  // Serial.print("Advertising started");
  // Serial.print("Bluetooth device active, waiting for connections...");
  pinMode(analogPin, INPUT); 
}

void loop() 
{
  static unsigned long preMillis = 0;
  
  // listen for BLE centrals devices
  BLEDevice central = BLE.central();

  // central device connected?
  if (central) 
  {
    digitalWrite(LEDB, LOW); // turn on the blue led
    digitalWrite(LEDR, HIGH);
    if(Serial_flag) Serial.print("Connected to central: ");
    if(Serial_flag) Serial.println(central.address()); // central device MAC address
    
    // while the central is still connected to peripheral:
    while (central.connected()) 
    {     
      unsigned long curMillis = millis(); //约50天才会溢出
      if (preMillis > curMillis) preMillis=0; // millis() rollover?
      if (curMillis - preMillis >= 10) // check values every 10mS---INTERVAL_MS=10
      {
        preMillis = curMillis;
        updateValues(); // call function for updating value to send to central
      }
      
    } // still here while central connected

    // central disconnected:
    digitalWrite(LEDB, HIGH);
    digitalWrite(LEDR, LOW);
    if(Serial_flag) Serial.print("Disconnected from central: ");
    if(Serial_flag) Serial.println(central.address());
  } // no central

  // unsigned long curMillis = millis(); //约50天才会溢出
  // if (preMillis > curMillis) preMillis=0; // millis() rollover?
  // if (curMillis - preMillis >= 10) // check values every 10mS---INTERVAL_MS=10
  // {
  //   preMillis = curMillis;
  //   updateValues(); // call function for updating value to send to central
  // }

}

void updateValues() 
{
  uint8_t averages=1; // average on this values count (accelerometer)
  
  // accelerometer averaged values/actual values
  static float ax=0;
  static float ay=0;
  static float az=0;
  static float gx=0;
  static float gy=0;
  static float gz=0;
  static float EMG_data=0;
  float ax1, ay1, az1, gx1, gy1, gz1, emg_data;
  static uint8_t i_a=0; // accelerometer readings counter
  
// read accelerometer values
  i_a++;
  ax1 = xIMU.readFloatAccelX();
  ay1 = xIMU.readFloatAccelY();
  az1 = xIMU.readFloatAccelZ();
  gx1 = xIMU.readFloatGyroX();// * CONVERT_G_TO_MS2;
  gy1 = xIMU.readFloatGyroY();// * CONVERT_G_TO_MS2;
  gz1 = xIMU.readFloatGyroZ();// * CONVERT_G_TO_MS2;
  
  emg_data = analogRead(analogPin);

  ax+=ax1;
  ay+=ay1;
  az+=az1;
  gx+=gx1;
  gy+=gy1;
  gz+=gz1;
  EMG_data += emg_data;
  if (i_a==averages) // send average over BLE
  {
    ax/=averages;
    ay/=averages;
    az/=averages;
    gx/=averages;
    gy/=averages;
    gz/=averages;
    EMG_data /= averages;
    if(Serial_flag) Serial.println("Accelerometer: "+String(ax)+","+String(ay)+","+String(az)+","+String(gx)+","+String(gy)+","+String(gz) + "," + String(EMG_data));//if(Serial_flag) 
    char messagebuf[50] = "";
    String message = "";
    String xdata = String(float(1*(ax)),3);
    String ydata = String(float(1*(ay)),3);
    String zdata = String(float(1*(az)),3);
    String xdata1 = String(float(1*(gx)),3);
    String ydata1 = String(float(1*(gy)),3);
    String zdata1 = String(float(1*(gz)),3);
    String emg_data1 = String(float(1*(EMG_data)),3);
    
    String SS=",";
    message = xdata+SS+ydata+SS+zdata+SS+xdata1+SS+ydata1+SS+zdata1+SS+emg_data1+SS;

    message.toCharArray(messagebuf, 200);
    chAccGyro.setValue((unsigned char*)messagebuf,200);

    ax=0;
    ay=0;
    az=0;
    gx=0;
    gy=0;
    gz=0;
    EMG_data = 0;
    i_a=0;
  }

  // Serial.println(a++);
  // if(a == 100) a =0;
}

void showID(char ID){
  int i = 0;
  while(i++, i <= device_id){ // FIXME
  delay(250);
  digitalWrite(LEDG, LOW); // led green on
  delay(250);
  digitalWrite(LEDG, HIGH); // led green off
  }
}

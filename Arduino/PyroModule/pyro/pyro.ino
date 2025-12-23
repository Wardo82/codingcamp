#include "analogreads.h"
#include "spicomm.h"

void setup() {
  Serial.begin(9600);

  pinMode(chipSelectPin, OUTPUT);
  SPI.beginTransaction(SPISettings(DRIVER_SPI_FREQUENCY, MSBFIRST, SPI_MODE2));
}

void loop() {
    float pins[NUM_OF_ANALOG_PINS] = {0.0};
    readAllAnalogPins(pins);
    printAllAnalogPins(pins);
    Serial.println();
    readAllDriverRegisters();
    delay(4000);
}

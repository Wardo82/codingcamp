/**
* SPI communication example code for a DRV8244S automotive driver.
* Datasheet: https://www.ti.com/lit/ds/symlink/drv8244-q1.pdf
*
* 1) The SCLK minium period is 100ns -> 10MHz max frequency.
* 2) The most significant bit (MSB) is shifted in and out first.
* 3) Driver works on SPI_MODE2
*   3.1) Data clock idle when SCS is high (SCLK and SDI pins are ignored, SDO to Hi-Z)
*   3.2) SDO data propagated on rising SCLK, SDI capture on edge SCLK
* 4) A full 16 SCLK cycles must occur for a valid transaction for a standard frame, or alternately,
*   for a daisy chain frame with "n" number of peripheral devices, 16 + (n x 16) SCLK cycles must
*   occur for a valid transaction. Else, a frame error (SPI_ERR) is reported and the data is ignored if it is a WRITE operation
*
* Pin outline:
* CSB: pin 7
* MOSI: pin 11
* MISO: pin 12
* SCK: pin 13
*
*/
#include <stdint.h>

#include <SPI.h>

#define DRIVER_SPI_FREQUENCY 10000000

// Set pin 10 as chip select for the digital port:
const int chipSelectPin = 10;

// drv8244_registers.h
#define DRV_REG_DEVICE_ID      0x00
#define DRV_REG_FAULT_SUMMARY  0x01
#define DRV_REG_STATUS1        0x02
#define DRV_REG_STATUS2        0x03
#define DRV_REG_COMMAND        0x08
#define DRV_REG_SPI_IN         0x09
#define DRV_REG_CONFIG1        0x0A
#define DRV_REG_CONFIG2        0x0B
#define DRV_REG_CONFIG3        0x0C
#define DRV_REG_CONFIG4        0x0D

// =====================
// Write Register
// =====================
static inline void drv_write(uint8_t address, uint8_t value)
{
    uint8_t tx[2];

    address &= 0x3F;                 // keep only 6 bits

    // Assemble SDI frame
    tx[0] = (0 << 7) |               // B15: frame type = 0
            (0 << 6) |               // B14: W0 = 0 (write)
            (address & 0x3F);        // B13..B8: address

    tx[1] = value;                   // data byte

    SPI.transfer(tx[0]);
    SPI.transfer(tx[1]);
}


// =====================
// Read Register
// =====================
static inline uint8_t drv_read(uint8_t address)
{
    uint8_t tx[2];
    uint8_t rx[2];

    address &= 0x3F;                 // only A5..A0

    // Assemble SDI read frame
    tx[0] = (0 << 7) |               // B15: frame type = 0
            (1 << 6) |               // B14: W0 = 1 (read)
            (address & 0x3F);        // B13..B8: address

    tx[1] = 0x00;                    // dummy data for read

    rx[0] = SPI.transfer(tx[0]);
    rx[1] = SPI.transfer(tx[1]);
    
    // rx[0] = status byte (11xxxxxx)
    // rx[1] = report byte (register value)
    return rx[1];
}

void readAllDriverRegisters() {
    uint8_t value = drv_read(DRV_REG_DEVICE_ID);
    Serial.print("DRV_REG_DEVICE_ID: ");
    Serial.print(value);
    Serial.println();
    
    value = drv_read(DRV_REG_FAULT_SUMMARY);
    Serial.print("DRV_REG_FAULT_SUMMARY: ");
    Serial.print(value);
    Serial.println();
    
    value = drv_read(DRV_REG_STATUS1);
    value = drv_read(DRV_REG_STATUS2);
    value = drv_read(DRV_REG_COMMAND);
    value = drv_read(DRV_REG_SPI_IN);
    value = drv_read(DRV_REG_CONFIG1);
    value = drv_read(DRV_REG_CONFIG2);
    value = drv_read(DRV_REG_CONFIG3);
    value = drv_read(DRV_REG_CONFIG4);
}

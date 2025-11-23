# Creating driver files for DRV8244-Q1: drv8244.h and drv8244.c
from pathlib import Path
h = r"""\
// drv8244.h
// Auto-generated driver header for DRV8244-Q1
// Public API: high-level structs + pack/unpack + read/write functions
// Uses raw-byte SPI transfer; user must provide spi_transfer implementation.
//
// Notes:
// - This header provides readable structs for registers.
// - Pack/unpack helpers convert structs <-> raw bytes.
// - All SPI transfers use drv_read_reg / drv_write_reg which are byte-accurate.
//
// Usage:
//   // Provide implementation of spi_transfer(tx, rx, len) elsewhere
//   uint8_t id = drv_read_reg(DRV_REG_DEVICE_ID);
//   drv_config1_t cfg = { .EN_OLA = 1, .VMOV_SEL = 2 };
//   drv_write_config1(cfg);
//   cfg = drv_read_config1();
//
// SPDX-License-Identifier: MIT

#ifndef DRV8244_H
#define DRV8244_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------
   Register addresses
   --------------------------- */
#define DRV_REG_DEVICE_ID       0x00u
#define DRV_REG_FAULT_SUMMARY   0x01u
#define DRV_REG_STATUS1         0x02u
#define DRV_REG_STATUS2         0x03u
#define DRV_REG_COMMAND         0x08u
#define DRV_REG_SPI_IN          0x09u
#define DRV_REG_CONFIG1         0x0Au
#define DRV_REG_CONFIG2         0x0Bu
#define DRV_REG_CONFIG3         0x0Cu
#define DRV_REG_CONFIG4         0x0Du

/* ---------------------------
   Low-level SPI interface
   --------------------------- */
/*
 * User must provide this function. It shall perform a full-duplex SPI transfer
 * of 'len' bytes. tx may be NULL for zeros; rx may be NULL to discard.
 */
extern void spi_transfer(uint8_t *tx, uint8_t *rx, uint8_t len);

/* Low-level register access (raw bytes) */
uint8_t drv_read_reg(uint8_t addr);
void drv_write_reg(uint8_t addr, uint8_t value);

/* ---------------------------
   Logical register structs
   --------------------------- */

/* DEVICE_ID (read-only) */
typedef struct {
    uint8_t DEV_ID : 6;    /* bits 7..2 */
    uint8_t REV_ID : 2;    /* bits 1..0 (REV_ID[1:0]) - note: stored in D7..D0 of register */
} drv_device_id_t;

/* FAULT_SUMMARY (read-only) */
typedef struct {
    uint8_t SPI_ERR : 1;   /* bit7 - SPI_ERR (OLA replaced by SPI_ERR in status) */
    uint8_t POR     : 1;   /* bit6 */
    uint8_t FAULT   : 1;   /* bit5 */
    uint8_t VMOV    : 1;   /* bit4 */
    uint8_t VMUV    : 1;   /* bit3 */
    uint8_t OCP     : 1;   /* bit2 */
    uint8_t TSD     : 1;   /* bit1 */
    uint8_t OLA     : 1;   /* bit0 */
} drv_fault_summary_t;

/* STATUS1 (read-only) */
typedef struct {
    uint8_t OLA1  : 1;  /* bit7 */
    uint8_t OLA2  : 1;  /* bit6 */
    uint8_t ITRIP_CMP : 1; /* bit5 */
    uint8_t ACTIVE   : 1; /* bit4 */
    uint8_t OCP_H1   : 1; /* bit3 */
    uint8_t OCP_L1   : 1; /* bit2 */
    uint8_t OCP_H2   : 1; /* bit1 */
    uint8_t OCP_L2   : 1; /* bit0 */
} drv_status1_t;

/* STATUS2 (read-only) - many bits N/A; represent full byte */
typedef uint8_t drv_status2_t;

/* COMMAND register (read/write) */
typedef struct {
    uint8_t CLR_FLT    : 1; /* bit7 */
    uint8_t reserved1  : 3; /* bits 6..4 N/A */
    uint8_t SPI_IN_LOCK1 : 1; /* bit3 (SPI_IN_LOCK[1]) */
    uint8_t SPI_IN_LOCK0 : 1; /* bit2 (SPI_IN_LOCK[0]) */
    uint8_t reserved0  : 1; /* bit1 N/A */
    uint8_t REG_LOCK0  : 1; /* bit0 (REG_LOCK[0]) */
} drv_command_t;

/* SPI_IN register (read/write) - reserved mapping kept as full byte */
typedef uint8_t drv_spi_in_t;

/* CONFIG1 (R/W) */
typedef struct {
    uint8_t EN_OLA      : 1; /* bit7 */
    uint8_t VMOV_SEL    : 2; /* bits6..5 */
    uint8_t SSC_DIS     : 1; /* bit4 */
    uint8_t OCP_RETRY   : 1; /* bit3 */
    uint8_t TSD_RETRY   : 1; /* bit2 */
    uint8_t VMOV_RETRY  : 1; /* bit1 */
    uint8_t OLA_RETRY   : 1; /* bit0 */
} drv_config1_t;

/* CONFIG2 (R/W) - PWM_EXTEND and S_DIAG bits in doc; represent as fields */
typedef struct {
    uint8_t PWM_EXTEND : 1; /* bit7 */
    uint8_t S_DIAG1    : 1; /* bit6 */
    uint8_t S_DIAG0    : 1; /* bit5 */
    uint8_t reserved   : 5; /* bits4..0 N/A or multi-bit items mapped elsewhere */
} drv_config2_t;

/* CONFIG3 (R/W) */
typedef struct {
    uint8_t TOFF1  : 1; /* bit7 */
    uint8_t TOFF0  : 1; /* bit6 */
    uint8_t reserved : 2; /* bits5..4 */
    uint8_t S_SR2  : 1; /* bit3 */
    uint8_t S_SR1  : 1; /* bit2 */
    uint8_t S_SR0  : 1; /* bit1 */
    uint8_t S_MODE : 1; /* bit0 */
} drv_config3_t;

/* CONFIG4 (R/W) */
typedef struct {
    uint8_t TOCP_SEL1 : 1; /* bit7 */
    uint8_t TOCP_SEL0 : 1; /* bit6 */
    uint8_t reserved  : 2; /* bits5..4 */
    uint8_t OCP_SEL1  : 1; /* bit3 */
    uint8_t OCP_SEL0  : 1; /* bit2 */
    uint8_t DRV0F_SEL : 1; /* bit1 - labeled DRV0F_SEL/DRV0F? use generic */
    uint8_t EN_IN1_SEL: 1; /* bit0 - EN_IN1_SEL / PH_IN2_SEL mapping */
} drv_config4_t;

/* ---------------------------
   Pack / Unpack helpers
   --------------------------- */

/* Pack struct -> byte */
uint8_t drv_pack_config1(drv_config1_t s);
drv_config1_t drv_unpack_config1(uint8_t raw);

uint8_t drv_pack_config2(drv_config2_t s);
drv_config2_t drv_unpack_config2(uint8_t raw);

uint8_t drv_pack_config3(drv_config3_t s);
drv_config3_t drv_unpack_config3(uint8_t raw);

uint8_t drv_pack_config4(drv_config4_t s);
drv_config4_t drv_unpack_config4(uint8_t raw);

/* Status/fault unpack */
drv_fault_summary_t drv_unpack_fault_summary(uint8_t raw);
drv_status1_t drv_unpack_status1(uint8_t raw);
drv_device_id_t drv_unpack_device_id(uint8_t raw);

/* ---------------------------
   High-level API
   --------------------------- */

/* CONFIG register read/write */
drv_config1_t drv_read_config1(void);
void drv_write_config1(drv_config1_t cfg);

drv_config2_t drv_read_config2(void);
void drv_write_config2(drv_config2_t cfg);

drv_config3_t drv_read_config3(void);
void drv_write_config3(drv_config3_t cfg);

drv_config4_t drv_read_config4(void);
void drv_write_config4(drv_config4_t cfg);

/* Read status / fault / device id */
drv_fault_summary_t drv_read_fault_summary(void);
drv_status1_t drv_read_status1(void);
drv_status2_t drv_read_status2(void);
drv_device_id_t drv_read_device_id(void);

/* COMMAND register */
drv_command_t drv_read_command(void);
void drv_write_command(drv_command_t cmd);

/* Utility */
void drv_clear_faults(void);

#ifdef __cplusplus
}
#endif

#endif /* DRV8244_H */

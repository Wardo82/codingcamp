// Implementation for drv8244.h
#include "drv8244.h"
#include <string.h>

/* ---------------------------
   Low-level frame assembly
   --------------------------- */

/*
 * SDI frame (2 bytes):
 * Byte0: [B15 frameType(0)] [B14 W0(read=1)] [B13..B8 addr(6)]
 * Byte1: data
 *
 * We'll assemble manually to guarantee bit positions.
 */

static void sdi_build(uint8_t addr, uint8_t is_read, uint8_t data, uint8_t tx[2])
{
    uint8_t cmd = 0u;
    uint8_t frame_type = 0u; /* standard frame = 0 */
    addr &= 0x3Fu; /* 6 bits */
    /* Bit packing: cmd = frame_type<<7 | W0<<6 | addr (bits5..0) */
    cmd = (uint8_t)((frame_type << 7) | ((is_read & 0x1u) << 6) | (addr & 0x3Fu));
    tx[0] = cmd;
    tx[1] = data;
}

uint8_t drv_read_reg(uint8_t addr)
{
    uint8_t tx[2];
    uint8_t rx[2];
    sdi_build(addr, 1u, 0x00u, tx);
    spi_transfer(tx, rx, 2u);
    /* rx[1] is report byte */
    return rx[1];
}

void drv_write_reg(uint8_t addr, uint8_t value)
{
    uint8_t tx[2];
    uint8_t rx[2];
    sdi_build(addr, 0u, value, tx);
    spi_transfer(tx, rx, 2u);
    (void)rx;
}

/* ---------------------------
   Pack / Unpack implementations
   --------------------------- */

uint8_t drv_pack_config1(drv_config1_t s)
{
    uint8_t v = 0u;
    v |= (uint8_t)((s.EN_OLA & 0x1u) << 7);
    v |= (uint8_t)((s.VMOV_SEL & 0x3u) << 5);
    v |= (uint8_t)((s.SSC_DIS & 0x1u) << 4);
    v |= (uint8_t)((s.OCP_RETRY & 0x1u) << 3);
    v |= (uint8_t)((s.TSD_RETRY & 0x1u) << 2);
    v |= (uint8_t)((s.VMOV_RETRY & 0x1u) << 1);
    v |= (uint8_t)((s.OLA_RETRY & 0x1u) << 0);
    return v;
}

drv_config1_t drv_unpack_config1(uint8_t raw)
{
    drv_config1_t s;
    s.EN_OLA = (uint8_t)((raw >> 7) & 0x1u);
    s.VMOV_SEL = (uint8_t)((raw >> 5) & 0x3u);
    s.SSC_DIS = (uint8_t)((raw >> 4) & 0x1u);
    s.OCP_RETRY = (uint8_t)((raw >> 3) & 0x1u);
    s.TSD_RETRY = (uint8_t)((raw >> 2) & 0x1u);
    s.VMOV_RETRY = (uint8_t)((raw >> 1) & 0x1u);
    s.OLA_RETRY = (uint8_t)((raw >> 0) & 0x1u);
    return s;
}

uint8_t drv_pack_config2(drv_config2_t s)
{
    uint8_t v = 0u;
    v |= (uint8_t)((s.PWM_EXTEND & 0x1u) << 7);
    v |= (uint8_t)((s.S_DIAG1 & 0x1u) << 6);
    v |= (uint8_t)((s.S_DIAG0 & 0x1u) << 5);
    /* reserved bits left as 0 */
    return v;
}

drv_config2_t drv_unpack_config2(uint8_t raw)
{
    drv_config2_t s;
    s.PWM_EXTEND = (uint8_t)((raw >> 7) & 0x1u);
    s.S_DIAG1 = (uint8_t)((raw >> 6) & 0x1u);
    s.S_DIAG0 = (uint8_t)((raw >> 5) & 0x1u);
    s.reserved = 0u;
    return s;
}

uint8_t drv_pack_config3(drv_config3_t s)
{
    uint8_t v = 0u;
    v |= (uint8_t)((s.TOFF1 & 0x1u) << 7);
    v |= (uint8_t)((s.TOFF0 & 0x1u) << 6);
    v |= (uint8_t)((s.S_SR2 & 0x1u) << 3);
    v |= (uint8_t)((s.S_SR1 & 0x1u) << 2);
    v |= (uint8_t)((s.S_SR0 & 0x1u) << 1);
    v |= (uint8_t)((s.S_MODE & 0x1u) << 0);
    return v;
}

drv_config3_t drv_unpack_config3(uint8_t raw)
{
    drv_config3_t s;
    s.TOFF1 = (uint8_t)((raw >> 7) & 0x1u);
    s.TOFF0 = (uint8_t)((raw >> 6) & 0x1u);
    s.reserved = 0u;
    s.S_SR2 = (uint8_t)((raw >> 3) & 0x1u);
    s.S_SR1 = (uint8_t)((raw >> 2) & 0x1u);
    s.S_SR0 = (uint8_t)((raw >> 1) & 0x1u);
    s.S_MODE = (uint8_t)((raw >> 0) & 0x1u);
    return s;
}

uint8_t drv_pack_config4(drv_config4_t s)
{
    uint8_t v = 0u;
    v |= (uint8_t)((s.TOCP_SEL1 & 0x1u) << 7);
    v |= (uint8_t)((s.TOCP_SEL0 & 0x1u) << 6);
    v |= (uint8_t)((s.OCP_SEL1 & 0x1u) << 3);
    v |= (uint8_t)((s.OCP_SEL0 & 0x1u) << 2);
    v |= (uint8_t)((s.DRV0F_SEL & 0x1u) << 1);
    v |= (uint8_t)((s.EN_IN1_SEL & 0x1u) << 0);
    return v;
}

drv_config4_t drv_unpack_config4(uint8_t raw)
{
    drv_config4_t s;
    s.TOCP_SEL1 = (uint8_t)((raw >> 7) & 0x1u);
    s.TOCP_SEL0 = (uint8_t)((raw >> 6) & 0x1u);
    s.reserved = 0u;
    s.OCP_SEL1 = (uint8_t)((raw >> 3) & 0x1u);
    s.OCP_SEL0 = (uint8_t)((raw >> 2) & 0x1u);
    s.DRV0F_SEL = (uint8_t)((raw >> 1) & 0x1u);
    s.EN_IN1_SEL = (uint8_t)((raw >> 0) & 0x1u);
    return s;
}

/* Status unpackers */
drv_fault_summary_t drv_unpack_fault_summary(uint8_t raw)
{
    drv_fault_summary_t s;
    s.SPI_ERR = (uint8_t)((raw >> 7) & 0x1u);
    s.POR     = (uint8_t)((raw >> 6) & 0x1u);
    s.FAULT   = (uint8_t)((raw >> 5) & 0x1u);
    s.VMOV    = (uint8_t)((raw >> 4) & 0x1u);
    s.VMUV    = (uint8_t)((raw >> 3) & 0x1u);
    s.OCP     = (uint8_t)((raw >> 2) & 0x1u);
    s.TSD     = (uint8_t)((raw >> 1) & 0x1u);
    s.OLA     = (uint8_t)((raw >> 0) & 0x1u);
    return s;
}

drv_status1_t drv_unpack_status1(uint8_t raw)
{
    drv_status1_t s;
    s.OLA1 = (uint8_t)((raw >> 7) & 0x1u);
    s.OLA2 = (uint8_t)((raw >> 6) & 0x1u);
    s.ITRIP_CMP = (uint8_t)((raw >> 5) & 0x1u);
    s.ACTIVE = (uint8_t)((raw >> 4) & 0x1u);
    s.OCP_H1 = (uint8_t)((raw >> 3) & 0x1u);
    s.OCP_L1 = (uint8_t)((raw >> 2) & 0x1u);
    s.OCP_H2 = (uint8_t)((raw >> 1) & 0x1u);
    s.OCP_L2 = (uint8_t)((raw >> 0) & 0x1u);
    return s;
}

drv_device_id_t drv_unpack_device_id(uint8_t raw)
{
    drv_device_id_t s;
    /* According to datasheet, DEV_ID mapped in bits 7..2 and REV_ID in bits1..0 */
    s.DEV_ID = (uint8_t)((raw >> 2) & 0x3Fu);
    s.REV_ID = (uint8_t)(raw & 0x03u);
    return s;
}

/* ---------------------------
   High-level API implementations
   --------------------------- */

drv_config1_t drv_read_config1(void)
{
    uint8_t raw = drv_read_reg(DRV_REG_CONFIG1);
    return drv_unpack_config1(raw);
}

void drv_write_config1(drv_config1_t cfg)
{
    uint8_t raw = drv_pack_config1(cfg);
    drv_write_reg(DRV_REG_CONFIG1, raw);
}

drv_config2_t drv_read_config2(void)
{
    uint8_t raw = drv_read_reg(DRV_REG_CONFIG2);
    return drv_unpack_config2(raw);
}

void drv_write_config2(drv_config2_t cfg)
{
    uint8_t raw = drv_pack_config2(cfg);
    drv_write_reg(DRV_REG_CONFIG2, raw);
}

drv_config3_t drv_read_config3(void)
{
    uint8_t raw = drv_read_reg(DRV_REG_CONFIG3);
    return drv_unpack_config3(raw);
}

void drv_write_config3(drv_config3_t cfg)
{
    uint8_t raw = drv_pack_config3(cfg);
    drv_write_reg(DRV_REG_CONFIG3, raw);
}

drv_config4_t drv_read_config4(void)
{
    uint8_t raw = drv_read_reg(DRV_REG_CONFIG4);
    return drv_unpack_config4(raw);
}

void drv_write_config4(drv_config4_t cfg)
{
    uint8_t raw = drv_pack_config4(cfg);
    drv_write_reg(DRV_REG_CONFIG4, raw);
}

drv_fault_summary_t drv_read_fault_summary(void)
{
    uint8_t raw = drv_read_reg(DRV_REG_FAULT_SUMMARY);
    return drv_unpack_fault_summary(raw);
}

drv_status1_t drv_read_status1(void)
{
    uint8_t raw = drv_read_reg(DRV_REG_STATUS1);
    return drv_unpack_status1(raw);
}

drv_status2_t drv_read_status2(void)
{
    return drv_read_reg(DRV_REG_STATUS2);
}

drv_device_id_t drv_read_device_id(void)
{
    uint8_t raw = drv_read_reg(DRV_REG_DEVICE_ID);
    return drv_unpack_device_id(raw);
}

drv_command_t drv_read_command(void)
{
    uint8_t raw = drv_read_reg(DRV_REG_COMMAND);
    drv_command_t s;
    s.CLR_FLT = (uint8_t)((raw >> 7) & 0x1u);
    s.reserved1 = 0u;
    s.SPI_IN_LOCK1 = (uint8_t)((raw >> 3) & 0x1u);
    s.SPI_IN_LOCK0 = (uint8_t)((raw >> 2) & 0x1u);
    s.reserved0 = 0u;
    s.REG_LOCK0 = (uint8_t)((raw >> 0) & 0x1u);
    return s;
}

void drv_write_command(drv_command_t cmd)
{
    uint8_t raw = 0u;
    raw |= (uint8_t)((cmd.CLR_FLT & 0x1u) << 7);
    raw |= (uint8_t)((cmd.SPI_IN_LOCK1 & 0x1u) << 3);
    raw |= (uint8_t)((cmd.SPI_IN_LOCK0 & 0x1u) << 2);
    raw |= (uint8_t)((cmd.REG_LOCK0 & 0x1u) << 0);
    drv_write_reg(DRV_REG_COMMAND, raw);
}

/* Utility: clear faults (sets CLR_FLT bit) */
void drv_clear_faults(void)
{
    drv_command_t c = {0};
    c.CLR_FLT = 1u;
    drv_write_command(c);
}

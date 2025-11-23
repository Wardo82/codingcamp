#define NUM_OF_ANALOG_PINS

void readAllAnalogPins(float pins[NUM_OF_ANALOG_PINS]) {

    // Read the continuity pins
    pins[0] = analogRead(A0) * (5.0 / 1023.0); // BSLN_CH1_CONT
    pins[1] = analogRead(A1) * (5.0 / 1023.0); // BSLN_CH2_CONT

    // Read driver's analog output
    pins[2] = analogRead(A2) * (5.0 / 1023.0); // DRV_IMON
    pins[3] = analogRead(A3) * (5.0 / 1023.0); // DRV_FAULT

    pins[5] = analogRead(A5) * (5.0 / 1023.0); // ARMED (NetBSL_36)
}

void printAllAnalogPins(float pins[NUM_OF_ANALOG_PINS]) {

    Serial.print("BSLN_CH1_CONT: ");
    Serial.print(pins[0]);
    Serial.println();

    Serial.print("BSLN_CH2_CONT: ");
    Serial.print(pins[1]);
    Serial.println();

    Serial.print("DRV_IMON: ");
    Serial.print(pins[2]);
    Serial.println();

    Serial.print("DRV_FAULT: ");
    Serial.print(pins[3]);
    Serial.println();

    ///

    Serial.print("ARMED: ");
    Serial.print(pins[5]);
    Serial.println();
}

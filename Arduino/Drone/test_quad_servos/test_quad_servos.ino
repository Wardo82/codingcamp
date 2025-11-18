#include <Servo.h>

// A Quadcopter has 4 servos
#define NUM_OF_SERVOS 4

Servo servos[NUM_OF_SERVOS];
const unsigned int SERVO_PINS[NUM_OF_SERVOS] = {6, 9, 10, 11};

void setSpeed(Servo* servo, int speed) {
  int angle = map(speed, 0, 100, 0, 180);
  servo->write(angle);
}

void setup() {

  for (int i = 0; i < NUM_OF_SERVOS; i++) {
    servos[i].attach(i);
    setSpeed(&servos[i], 0);
  }

}

void loop() {
  for (size_t i = 0; i < NUM_OF_SERVOS; i++) {
    printf("Testing Servo %i attached to PIN %i\n", i, SERVO_PINS[i]);
    Servo* servo = &servos[i];
    int speed;
    for (speed = 0; speed <= 70; speed += 5) {
      //Cycles speed up to 70% power for 1 second
      setSpeed(servo, speed);
      //Creates variable for speed to be used in in for loop
      delay(1000);
    }
    delay(4000); //Stays on for 4 seconds
    for(speed = 70; speed > 0; speed -= 5) {
      // Cycles speed down to 0% power for 1 second
      setSpeed(servo, speed);
      delay(1000);
    }
    //Sets speed variable to zero no matter what
    setSpeed(servo, 0);
    //Turns off for 1 second
    delay(1000);
  }
}

# Codigos correctores Reed-Solomon "de abajo a arriba"

Esta carpeta contiene codigo para entender y explorar codigos Reed-Solomon desde su fundamento matematico en polinomios hasta su uso en codigos correctores de errores.

Todo parte de este gran articulo [Reed-Solomon Error Correcting Codes from the bottom up](https://tomverbeure.github.io/2022/08/07/Reed-Solomon.html). En esta carpeta tomare las ideas de ahi y las implemento en Python.

La meta en ensenarme a mi mismo, si le sirve a alguien mas, mejor.

## Introduccion

Algunos comentarios relevantes del articulo:

    ...
    La corrección de errores hacia adelante (FEC) Reed-Solomon es uno de estos métodos de codificación. Hasta el descubrimiento de técnicas de codificación más avanzadas, como los códigos turbo y los códigos de comprobación de paridad de baja densidad (LDPC), era una de las formas más eficaces de proteger el almacenamiento y la transmisión de datos contra errores: las naves espaciales Voyager utilizaron la codificación Reed-Solomon para transmitir imágenes cuando se encontraban entre Saturno y Urano, y los CD pueden recuperarse de arañazos que corrompen hasta 4000 bits gracias al uso inteligente de no uno, sino dos códigos Reed-Solomon.
    ...

Sobre polinomios. Una de las caracteristicas mas importantes de los polinomios:

    ...
    Cualquier funcion polinomica de grado n-1 esta unicamente definida por n puntos cualesquiera que yacen en ella.
    ...

## Lista de scripts

1. Polinomios y como funciona un algebra de polinomios.
2. 
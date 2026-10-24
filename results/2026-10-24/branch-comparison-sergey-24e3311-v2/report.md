# Technical comparison attempt

Strict bitwise float truth equality failed on Sergey Mac CSV parsing: max delta5.684341886080802e-14 over104rows. No model fit, no metric conclusion. New v3 uses explicit atol1e-10 for numeric targets; classification remains exact.

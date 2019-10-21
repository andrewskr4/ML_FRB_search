void acqch1fft_help()
{
  puts("");
  puts("acqch1fft - convert raw data to fiterbank format");
  puts("Usage - acqch1fft <input file> _{options} ");
  puts("\noptions:\n");
  puts("-o filename - output file containing filterbank data (def=stdout)");
  puts("-ts sampling time (sec) (default=0.000064, nacc=1)");
  puts("-nacc number of spectra to accumalate (default=1,ts=nacc*0.000064)");
  puts("-mjd Modiefied Julian Date for the start of observations");
  puts("");
  puts("");

  puts("");
}

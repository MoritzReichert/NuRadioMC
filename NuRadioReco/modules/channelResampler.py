import numpy as np
import fractions
from scipy import signal
from decimal import Decimal
import copy
from NuRadioReco.utilities import units, fft
import logging
logger = logging.getLogger('channelResampler')


class channelResampler:
    """
    Resamples the trace to a new sampling rate.
    """

    def __init__(self):
        self.begin()

    def begin(self, debug=False):
        self.__max_upsampling_factor = 5000
        self.__debug = debug

        """
        Begin the channelResampler

        Parameters
        ---------

        __debug: bool
            Debug switch

        """


    def run(self, evt, station, det, sampling_rate):
        """
        Run the channelResampler

        Parameters
        ---------

        evt, station, det
            Event, Station, Detector
        sampling_rate: float
            In units 1/time provides the desired sampling rate of the data.

        """
        channels = station.get_channels()

        orig_binning = 1. / channels[0].get_sampling_rate()  # assume that all channels have the same sampling rate
        target_binning = 1. / sampling_rate
        resampling_factor = fractions.Fraction(Decimal(orig_binning / target_binning)).limit_denominator(self.__max_upsampling_factor)
        if resampling_factor == self.__max_upsampling_factor:
            logger.warning("Safeguard caught, max upsampling {} factor reached.".format(self.__max_upsampling_factor))
        logger.debug("resampling channel trace by {}. Original binning is {:.3g} ns, target binning is {:.3g} ns".format(resampling_factor,
                                                                                                                         orig_binning / units.ns,
                                                                                                                         target_binning / units.ns))
        for channel in channels:
            trace = channel.get_trace()
            if(self.__debug):
                ff = np.fft.rfftfreq(len(trace), orig_binning)
                spec = fft.time2freq(trace)
                spec[ff >= 500 * units.MHz] = 0
                trace = fft.freq2time(spec)
                trace_old = copy.copy(trace)

            if(resampling_factor.numerator != 1):
                trace = signal.resample(trace, resampling_factor.numerator * len(trace))  # , window='hann')
            if(resampling_factor.denominator != 1):
                trace = signal.resample(trace, len(trace) / resampling_factor.denominator)  # , window='hann')

            if(self.__debug):

                if(resampling_factor.denominator != 1):
                    trace2 = signal.resample(trace, resampling_factor.denominator * len(trace))  # , window='hann')
                if(resampling_factor.numerator != 1):
                    trace2 = signal.resample(trace, len(trace) / resampling_factor.numerator)  # , window='hann')

                import matplotlib.pyplot as plt
                plt.plot(np.fft.rfftfreq(len(trace_old), orig_binning), np.abs(fft.time2freq(trace_old)))
                plt.plot(np.fft.rfftfreq(len(trace2), orig_binning), np.abs(fft.time2freq(trace2)))
                plt.show()

                tt = np.arange(0, len(trace_old) * orig_binning, orig_binning)
                plt.plot(tt, trace_old, '.-')
                tt = np.arange(0, len(trace2) * orig_binning, orig_binning)
                plt.plot(tt, trace2, '.-')
                tt = np.arange(0, len(trace) / sampling_rate, 1. / sampling_rate)
                plt.plot(tt, trace, 'o-')
                plt.show()

            channel.set_trace(trace, 1. / target_binning)

    def end(self):
        pass

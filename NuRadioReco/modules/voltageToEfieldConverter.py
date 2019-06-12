import numpy as np
import os
import copy
from NuRadioReco.utilities import geometryUtilities as geo_utl
from NuRadioReco.utilities import units, fft
from NuRadioReco.utilities import ice
from NuRadioReco.detector import antennapattern
from NuRadioReco.utilities import trace_utilities
import NuRadioReco.framework.base_trace
import NuRadioReco.framework.electric_field
import matplotlib.pyplot as plt
from scipy import signal

import logging
logger = logging.getLogger('voltageToEfieldConverter')

from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.framework.parameters import electricFieldParameters as efp

def get_array_of_channels(station, use_channels, det, zenith, azimuth,
                          antenna_pattern_provider, time_domain=False):
    time_shifts = np.zeros(len(use_channels))
    t_cables = np.zeros(len(use_channels))
    t_geos = np.zeros(len(use_channels))

    station_id = station.get_id()
    site = det.get_site(station_id)
    n_ice = ice.get_refractive_index(-0.01, site)
    for iCh, channel in enumerate(station.iter_channels(use_channels)):
        channel_id = channel.get_id()

        antenna_position = det.get_relative_position(station_id, channel_id)
        # determine refractive index of signal propagation speed between antennas
        refractive_index = ice.get_refractive_index(1, site)  # if signal comes from above, in-air propagation speed
        if(zenith > 0.5 * np.pi):
            refractive_index = ice.get_refractive_index(antenna_position[2], site)  # if signal comes from below, use refractivity at antenna position
        time_shift = -geo_utl.get_time_delay_from_direction(zenith, azimuth, antenna_position, n=refractive_index)
        t_geos[iCh] = time_shift
        t_cables[iCh] = channel.get_trace_start_time()
        logger.debug("time shift channel {}: {:.2f}ns (signal prop), {:.2f}ns (trace start time)".format(channel.get_id(), time_shift, channel.get_trace_start_time()))
        time_shift += channel.get_trace_start_time()
        time_shifts[iCh] = time_shift

    delta_t = time_shifts.max() - time_shifts.min()
    tmin = time_shifts.min()
    tmax = time_shifts.max()
    logger.debug("adding relative station time = {:.0f}ns".format((t_cables.min() + t_geos.max()) / units.ns))
    logger.debug("delta t is {:.2f}".format(delta_t / units.ns))
    trace_length = station.get_channel(0).get_times()[-1] - station.get_channel(0).get_times()[0]
    debug_cut = 0
    if(debug_cut):
        fig, ax = plt.subplots(len(use_channels), 1)

    traces = []
    n_samples = None
    for iCh, channel in enumerate(station.iter_channels(use_channels)):
        tstart = delta_t - (time_shifts[iCh] - tmin)
        tstop = tmax - time_shifts[iCh] - delta_t + trace_length
        iStart = int(round(tstart * channel.get_sampling_rate()))
        iStop = int(round(tstop * channel.get_sampling_rate()))
        if(n_samples is None):
            n_samples = iStop - iStart
            if(n_samples % 2):
                n_samples -= 1

        trace = copy.copy(channel.get_trace())  # copy to not modify data structure
        trace = trace[iStart:(iStart + n_samples)]
        if(debug_cut):
            ax[iCh].plot(trace)
        base_trace = NuRadioReco.framework.base_trace.BaseTrace()  # create base trace class to do the fft with correct normalization etc.
        base_trace.set_trace(trace, channel.get_sampling_rate())
        traces.append(base_trace)
    times = traces[0].get_times()  # assumes that all channels have the same sampling rate
    if(time_domain):  # save time domain traces first to avoid extra fft
        V_timedomain = np.zeros((len(use_channels), len(times)))
        for iCh, trace in enumerate(traces):
            V_timedomain[iCh] = trace.get_trace()
    frequencies = traces[0].get_frequencies()  # assumes that all channels have the same sampling rate
    V = np.zeros((len(use_channels), len(frequencies)), dtype=np.complex)
    for iCh, trace in enumerate(traces):
        V[iCh] = trace.get_frequency_spectrum()

    efield_antenna_factor = trace_utilities.get_efield_antenna_factor(station, frequencies, use_channels, det, zenith, azimuth, antenna_pattern_provider)
    
    if(debug_cut):
        plt.show()

    if(time_domain):
        return efield_antenna_factor, V, V_timedomain

    return efield_antenna_factor, V


def stacked_lstsq(L, b, rcond=1e-10):
    """
    Solve L x = b, via SVD least squares cutting of small singular values
    L is an array of shape (..., M, N) and b of shape (..., M).
    Returns x of shape (..., N)
    """
    u, s, v = np.linalg.svd(L, full_matrices=False)
    s_max = s.max(axis=-1, keepdims=True)
    s_min = rcond * s_max
    inv_s = np.zeros_like(s)
    inv_s[s >= s_min] = 1 / s[s >= s_min]
    x = np.einsum('...ji,...j->...i', v,
                  inv_s * np.einsum('...ji,...j->...i', u, b.conj()))
    return np.conj(x, x)


class voltageToEfieldConverter:

    def __init__(self):
        self.begin()

    def begin(self):
        self.antenna_provider = antennapattern.AntennaPatternProvider()
        pass

    def run(self, evt, station, det, debug=False, debug_plotpath=None, use_channels=[0, 1, 2, 3], use_MC_direction=False):
        """
        run method. This function is executed for each event

        Parameters
        ---------
        evt
        station
        det
        debug: bool
            if True debug plotting is enables
        debug_plotpath: string or None
            if not None plots will be saved to a file rather then shown. Plots will
            be save into the `debug_plotpath` directory
        use_channels: array of ints
            the channel ids to use for the electric field reconstruction
        """
        event_time = station.get_station_time()
        station_id = station.get_id()

        if use_MC_direction:
            zenith = station.get_sim_station()[stnp.zenith]
            azimuth = station.get_sim_station()[stnp.azimuth]
            sim_present = True
        else:
            logger.info("Using reconstructed (or starting) angles as no signal arrival angles are present")
            zenith = station[stnp.zenith]
            azimuth = station[stnp.azimuth]
            sim_present = False


        efield_antenna_factor, V = get_array_of_channels(station, use_channels, det, zenith, azimuth, self.antenna_provider)
        n_frequencies = len(V[0])
        denom = (efield_antenna_factor[0][0] * efield_antenna_factor[1][1] - efield_antenna_factor[0][1] * efield_antenna_factor[1][0])
        mask = np.abs(denom) != 0
        # solving for electric field using just two orthorgonal antennas
        E1 = np.zeros_like(V[0])
        E2 = np.zeros_like(V[0])
        E1[mask] = (V[0] * efield_antenna_factor[1][1] - V[1] * efield_antenna_factor[0][1])[mask] / denom[mask]
        E2[mask] = (V[1] - efield_antenna_factor[1][0] * E1)[mask] / efield_antenna_factor[1][1][mask]
        denom = (efield_antenna_factor[0][0] * efield_antenna_factor[-1][1] - efield_antenna_factor[0][1] * efield_antenna_factor[-1][0])
        mask = np.abs(denom) != 0
        E1[mask] = (V[0] * efield_antenna_factor[-1][1] - V[-1] * efield_antenna_factor[0][1])[mask] / denom[mask]
        E2[mask] = (V[-1] - efield_antenna_factor[-1][0] * E1)[mask] / efield_antenna_factor[-1][1][mask]
        # solve it in a vectorized way
        efield3_f = np.zeros((2, n_frequencies), dtype=np.complex)
        efield3_f[:, mask] = np.moveaxis(stacked_lstsq(np.moveaxis(efield_antenna_factor[:, :, mask], 2, 0), np.moveaxis(V[:, mask], 1, 0)), 0, 1)
        # add eR direction
        efield3_f = np.array([np.zeros_like(efield3_f[0], dtype=np.complex),
                             efield3_f[0],
                             efield3_f[1]])

        electric_field = NuRadioReco.framework.electric_field.ElectricField(use_channels, [0,0,0])
        electric_field.set_frequency_spectrum(efield3_f, station.get_channel(0).get_sampling_rate())
        electric_field.set_parameter(efp.zenith, zenith)
        electric_field.set_parameter(efp.azimuth, azimuth)
        #figure out the timing of the E-field
        t_shifts = np.zeros(V.shape[0])
        site = det.get_site(station_id)
        if(station.get_parameter(stnp.zenith) > 0.5 * np.pi):
            refractive_index = ice.get_refractive_index(antenna_position[2], site)  # if signal comes from below, use refractivity at antenna position
        else:
            refractive_index = ice.get_refractive_index(1, site)  # if signal comes from above, in-air propagation speed
        for i_ch, channel_id in enumerate(use_channels):
            antenna_position = det.get_relative_position(station.get_id(), channel_id)
            t_shifts[i_ch] = station.get_channel(channel_id).get_trace_start_time() -geo_utl.get_time_delay_from_direction(zenith, azimuth, antenna_position, n=refractive_index)
        
        electric_field.set_trace_start_time(t_shifts.max())
        station.add_electric_field(electric_field)

        if debug:
            fig, (ax2, ax2f) = plt.subplots(2, 1, figsize=(10, 8))
            lw = 2
            times = station.get_times() / units.ns
            ax2.plot(times, station.get_trace()[1] / units.mV * units.m, "-C0", label="reconstructed eTheta", lw=lw)
            ax2.plot(times, station.get_trace()[2] / units.mV * units.m, "-C1", label="reconstructed ePhi", lw=lw)
            ax2.set_xlim(400, 600)
            ff = station.get_frequencies() / units.MHz
            ax2f.plot(ff[ff < 500], np.abs(station.get_frequency_spectrum()[1][ff < 500]) / units.mV * units.m, "-C0", label="4 stations lsqr eTheta", lw=lw)
            ax2f.plot(ff[ff < 500], np.abs(station.get_frequency_spectrum()[2][ff < 500]) / units.mV * units.m, "-C1", label="4 stations lsqr ePhi", lw=lw)

            if station.has_sim_station():
                sim_station = station.get_sim_station()
                logger.debug("station start time {:.1f}ns, relativ sim station time = {:.1f}".format(station.get_trace_start_time(), sim_station.get_trace_start_time()))
                ax2.plot(sim_station.get_times() / units.ns, sim_station.get_trace()[1] / units.mV * units.m, "--C2", label="simulation eTheta", lw=lw)
                ax2.plot(sim_station.get_times() / units.ns, sim_station.get_trace()[2] / units.mV * units.m, "--C3", label="simulation ePhi", lw=lw)
                ax2f.plot(sim_station.get_frequencies() / units.MHz, np.abs(sim_station.get_frequency_spectrum()[1] / units.mV * units.m), "--C2", label="simulation eTheta", lw=lw)
                ax2f.plot(sim_station.get_frequencies() / units.MHz, np.abs(sim_station.get_frequency_spectrum()[2] / units.mV * units.m), "--C3", label="simulation ePhi", lw=lw)

            ax2.legend(fontsize="xx-small")
            ax2.set_xlabel("time [ns]")
            ax2.set_ylabel("electric-field [mV/m]")
            ax2f.set_ylim(1e-3, 5)
            ax2f.set_xlabel("Frequency [MHz]")
            ax2f.set_xlim(100, 500)
            ax2f.semilogy(True)
            if sim_present:
                sim = station.get_sim_station()
                fig.suptitle("Simulation: Zenith {:.1f}, Azimuth {:.1f}".format(np.rad2deg(sim[stnp.zenith]), np.rad2deg(sim[stnp.azimuth])))
            else:
                fig.suptitle("Data: reconstructed zenith {:.1f}, azimuth {:.1f}".format(np.rad2deg(zenith), np.rad2deg(azimuth)))
            fig.tight_layout()
            fig.subplots_adjust(top=0.95)
            if(debug_plotpath is not None):
                fig.savefig(os.path.join(debug_plotpath, 'run_{:05d}_event_{:06d}_efield.png'.format(evt.get_run_number(), evt.get_id())))
                plt.close(fig)

            # plot antenna response and channels
            fig, ax = plt.subplots(len(V), 2, sharex='col', sharey='col')
            for iCh in range(len(V)):
                ax[iCh, 0].plot(ff, np.abs(efield_antenna_factor[iCh][0]), label="theta, channel {}".format(use_channels[iCh]), lw=lw)
                ax[iCh, 0].plot(ff, np.abs(efield_antenna_factor[iCh][1]), label="phi, channel {}".format(use_channels[iCh]), lw=lw)
                ax[iCh, 0].legend(fontsize='xx-small')
                ax[iCh, 0].set_xlim(0, 500)
                ax[iCh, 1].set_xlim(400, 600)
                ax[iCh, 1].plot(times, fft.freq2time(V[iCh]) / units.micro / units.V, lw=lw)
                ax[iCh, 0].set_ylabel("H [m]")
                ax[iCh, 1].set_ylabel(r"V [$\mu$V]")
                RMS = det.get_noise_RMS(station.get_id(), 0)
                ax[iCh, 1].text(0.6, 0.8, 'S/N={:.1f}'.format(np.max(np.abs(fft.freq2time(V[iCh])) / RMS)), transform=ax[iCh, 1].transAxes)
            ax[-1, 1].set_xlabel("time [ns]")
            ax[-1, 0].set_xlabel("frequency [MHz]")
            fig.tight_layout()
            if(debug_plotpath is not None):
                fig.savefig(os.path.join(debug_plotpath, 'run_{:05d}_event_{:06d}_channels.png'.format(evt.get_run_number(), evt.get_id())))
                plt.close(fig)

    def end(self):
        pass

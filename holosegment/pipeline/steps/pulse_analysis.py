from holosegment.pipeline.step import BaseStep, NestedStep
from holosegment.segmentation import pulse_analysis

class PulseAnalysisStep(NestedStep):
    requires = ["M0_ff_video", "retinal_vessel_mask"]
    produces = ["correlation", "diasys_image"]
    name = "pulse_analysis"

    def __init__(self):
        self.substeps = [
            PreArteryMaskStep(),
            ComputeTemporalCuesStep()
        ]

    def run(self, ctx):
        for step in self.substeps:
            step.run(ctx)
            
class PreArteryMaskStep(BaseStep):
    requires = ["M0_ff_video", "retinal_vessel_mask", "optic_disc_center"]
    produces = ["pre_artery_mask"]
    name = "pre_artery_mask"

    def _relevant_config(self, ctx):
        return {"fs": ctx.holodoppler_config["fs"]}

    def run(self, ctx):
        video = ctx.cache["M0_ff_video"]
        vessel_mask = ctx.cache["retinal_vessel_mask"]

        sampling_frequency = ctx.holodoppler_config["fs"]

        pre_artery_mask, pre_vein_mask = pulse_analysis.compute_pre_artery_mask(video, vessel_mask, ctx.cache["optic_disc_center"], sampling_frequency, ctx.output_manager)
        ctx.output_manager.save(self.name, "pre_artery_mask", pre_artery_mask, "png")
        ctx.cache["pre_artery_mask"] = pre_artery_mask
        ctx.output_manager.save(self.name, "pre_vein_mask", pre_vein_mask, "png")
        ctx.cache["pre_vein_mask"] = pre_vein_mask

class ComputeTemporalCuesStep(BaseStep):
    requires = ["M0_ff_video", "pre_artery_mask", "choroidal_vessel_mask"]
    produces = ["correlation", "diasys_image"]
    name = "temporal_cues"

    def _relevant_config(self, ctx):
        return {"fs": ctx.holodoppler_config["fs"],
                "batch_stride": ctx.holodoppler_config["batch_stride"]}

    def run(self, ctx):
        video = ctx.require("M0_ff_video")
        pre_artery_mask = ctx.require("pre_artery_mask")
        choroidal_vessel_mask = ctx.require("choroidal_vessel_mask")

        correlation, pulse = pulse_analysis.compute_correlation(video, pre_artery_mask)
        correlation_choroidal, pulse_choroidal = pulse_analysis.compute_correlation(video, choroidal_vessel_mask)
        ctx.set("correlation", correlation)
        ctx.output_manager.save(self.name, "correlation_map", correlation, "png")
        ctx.output_manager.save(self.name, "choroidal_correlation_map", correlation_choroidal, "png")
        ctx.output_manager.save_plot(self.name, "pulse", pulse, "Pulse (correlation)")
        ctx.output_manager.save_plot(self.name, "pulse_choroidal", pulse_choroidal, "Pulse Choroidal (correlation)")

        sampling_frequency = ctx.holodoppler_config["fs"]
        stride = ctx.holodoppler_config["batch_stride"]

        diasys, M0_Systole_img, M0_Diastole_img, fullPulse = pulse_analysis.compute_diasys_image(video, pre_artery_mask, sampling_frequency=sampling_frequency, stride=stride)
        choroid_diasys, choroid_systole, choroid_diastole, choroid_fullPulse = pulse_analysis.compute_diasys_image(video, choroidal_vessel_mask, sampling_frequency=sampling_frequency, stride=stride)
        ctx.output_manager.save(self.name, "diasys_image", diasys, "png")
        ctx.output_manager.save(self.name, "M0_Systole_img", M0_Systole_img, "png")
        ctx.output_manager.save(self.name, "M0_Diastole_img", M0_Diastole_img, "png")
        ctx.output_manager.save_plot(self.name, "fullPulse", fullPulse, title = "Full Pulse Analysis (mean intensity over time)")

        ctx.output_manager.save(self.name, "choroid_diasys_image", choroid_diasys, "png")
        ctx.output_manager.save(self.name, "choroid_systole_img", choroid_systole, "png")
        ctx.output_manager.save(self.name, "choroid_diastole_img", choroid_diastole, "png")
        ctx.output_manager.save_plot(self.name, "choroid_fullPulse", choroid_fullPulse, title = "Choroidal Full Pulse Analysis (mean intensity over time)")

        ctx.set("diasys_image", diasys)

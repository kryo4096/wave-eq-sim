use std::vec::Vec;
use std::{error::Error, time::Instant};
use std::{sync::Arc, thread, time::Duration};

use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, DynamicState, SubpassContents},
    descriptor::{descriptor_set::PersistentDescriptorSet, PipelineLayoutAbstract},
    device::{Device, DeviceExtensions, Features},
    format::Format,
    framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass},
    image::{Dimensions, ImageUsage, ImageViewAccess, StorageImage, SwapchainImage},
    instance::{Instance, PhysicalDevice},
    pipeline::{viewport::Viewport, ComputePipeline, GraphicsPipeline},
    sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode},
    swapchain::{
        self, AcquireError, ColorSpace, FullscreenExclusive, PresentMode, SurfaceTransform,
        Swapchain, SwapchainCreationError,
    },
    sync::{self, FlushError, GpuFuture},
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    dpi::PhysicalSize,
    event::{ElementState, Event, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

mod cs {
    const _RECOMPILE_DUMMY: &str = include_str!("../res/shaders/compute.comp.glsl");

    vulkano_shaders::shader! {
        ty: "compute",
        path: "res/shaders/compute.comp.glsl"
    }
}

mod vs {
    const _RECOMPILE_DUMMY: &str = include_str!("../res/shaders/vertex.vert.glsl");

    vulkano_shaders::shader! {
        ty: "vertex",
        path: "res/shaders/vertex.vert.glsl"
    }
}

mod fs {
    const _RECOMPILE_DUMMY: &str = include_str!("../res/shaders/fragment.frag.glsl");

    vulkano_shaders::shader! {
        ty: "fragment",
        path: "res/shaders/fragment.frag.glsl"
    }
}

#[derive(Default, Copy, Clone)]
struct Vertex {
    position: [f32; 2],
}

impl Vertex {
    const fn new(x: f32, y: f32) -> Self {
        Self { position: [x, y] }
    }
}

vulkano::impl_vertex!(Vertex, position);

const SCREEN_QUAD: [Vertex; 6] = [
    Vertex::new(-1., -1.),
    Vertex::new(-1., 1.),
    Vertex::new(1., 1.),
    Vertex::new(-1., -1.),
    Vertex::new(1., -1.),
    Vertex::new(1., 1.),
];

fn main() -> Result<(), Box<dyn Error>> {
    let instance = {
        let extensions = vulkano_win::required_extensions();
        Instance::new(None, &extensions, None).expect("failed to create instance")
    };

    let physical = PhysicalDevice::enumerate(&instance)
        .next()
        .expect("no device available");

    let events_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .with_inner_size(PhysicalSize::new(1024, 1024))
        .with_resizable(false)
        .with_title("Wave Equation (Click and Drag to apply force to pixels)")
        .build_vk_surface(&events_loop, instance.clone())?;

    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
        .expect("couldn't find a graphical queue family");

    let (device, mut queues) = Device::new(
        physical,
        &Features {
            shader_storage_image_extended_formats: true,
            ..Features::none()
        },
        &DeviceExtensions {
            khr_storage_buffer_storage_class: true,
            khr_swapchain: true,
            ..DeviceExtensions::none()
        },
        std::array::IntoIter::new([(queue_family, 0.5)]),
    )
    .expect("failed to create device");

    let caps = surface
        .capabilities(physical)
        .expect("failed to get surface capabilities");

    let queue = queues.next().ok_or("failed to get queue")?;

    let dimensions: [u32; 2] = surface.window().inner_size().into();
    let (mut swapchain, images) = {
        let alpha = caps
            .supported_composite_alpha
            .iter()
            .next()
            .ok_or("failed to get alpha channel")?;
        let format = caps.supported_formats[0].0;

        Swapchain::new(
            device.clone(),
            surface.clone(),
            caps.min_image_count,
            format,
            dimensions,
            1,
            ImageUsage::color_attachment(),
            &queue,
            SurfaceTransform::Identity,
            alpha,
            PresentMode::Fifo,
            FullscreenExclusive::Default,
            true,
            ColorSpace::SrgbNonLinear,
        )
        .expect("failed to create swapchain")
    };

    let vertex_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        std::array::IntoIter::new(SCREEN_QUAD),
    )?;

    let cs = cs::Shader::load(device.clone()).expect("failed to create shader");

    let compute_pipeline = Arc::new(
        ComputePipeline::new(device.clone(), &cs.main_entry_point(), &(), None)
            .expect("failed to create compute pipeline"),
    );

    let compute_layout = compute_pipeline
        .layout()
        .descriptor_set_layout(0)
        .ok_or("unable to get compute layout")?;

    let image = StorageImage::with_usage(
        device.clone(),
        Dimensions::Dim2d {
            width: dimensions[0] / 4,
            height: dimensions[1] / 4,
        },
        Format::R32G32Sfloat,
        ImageUsage {
            sampled: true,
            storage: true,
            ..ImageUsage::none()
        },
        Some(queue.family()),
    )?;

    let sampler = Sampler::new(
        device.clone(),
        Filter::Nearest,
        Filter::Nearest,
        MipmapMode::Nearest,
        SamplerAddressMode::Repeat,
        SamplerAddressMode::Repeat,
        SamplerAddressMode::Repeat,
        0.0,
        1.0,
        0.0,
        0.0,
    )?;

    let compute_set = Arc::new(
        PersistentDescriptorSet::start(compute_layout.clone())
            .add_image(image.clone())?
            .build()?,
    );

    let vertex_shader = vs::Shader::load(device.clone()).expect("failed to create vertex shader");
    let fragment_shader =
        fs::Shader::load(device.clone()).expect("failed to create fragment shader");

    let render_pass = Arc::new(vulkano::single_pass_renderpass!(device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.format(),
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    )?);

    let pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vertex_shader.main_entry_point(), ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fragment_shader.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())?,
    );

    let render_layout = pipeline.layout().descriptor_set_layout(0).unwrap();

    let render_set = Arc::new(
        PersistentDescriptorSet::start(render_layout.clone())
            .add_sampled_image(image.clone(), sampler.clone())?
            .build()?,
    );

    let mut dynamic_state = DynamicState::none();

    let mut framebuffers =
        window_size_dependent_setup(&images, render_pass.clone(), &mut dynamic_state);

    let mut recreate_swapchain = true;

    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    let mut last_frame_time = Instant::now();

    let mut init_image = true;

    let mut mouse_pressed = false;

    let mut mouse_pos = [0., 0.];
    let mut wave_pos = [0., 0.];

    events_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: winit::event::WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(_),
            ..
        } => {
            recreate_swapchain = true;
        }
        Event::WindowEvent {
            event: WindowEvent::MouseInput { state, button, .. },
            ..
        } => {
            if button == MouseButton::Left {
                match state {
                    ElementState::Pressed => {
                        mouse_pressed = true;
                        wave_pos = mouse_pos;
                    }
                    ElementState::Released => mouse_pressed = false,
                }
            }
        }
        Event::WindowEvent {
            event: WindowEvent::CursorMoved { position, .. },
            ..
        } => {
            mouse_pos = [
                position.x as f32 / dimensions[0] as f32,
                position.y as f32 / dimensions[1] as f32,
            ]
        }
        Event::WindowEvent {
            event:
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            virtual_keycode,
                            state,
                            ..
                        },
                    ..
                },
            ..
        } => {
            if virtual_keycode == Some(VirtualKeyCode::R) && state == ElementState::Pressed {
                init_image = true;
            }
        }
        Event::RedrawEventsCleared => {
            previous_frame_end.as_mut().unwrap().cleanup_finished();

            let delta_time = Instant::now() - last_frame_time;
            last_frame_time = Instant::now();

            if recreate_swapchain {
                let dimensions: [u32; 2] = surface.window().inner_size().into();

                let (new_swapchain, new_images) =
                    match swapchain.recreate_with_dimensions(dimensions) {
                        Ok(r) => r,
                        Err(SwapchainCreationError::UnsupportedDimensions) => return,
                        Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                    };

                swapchain = new_swapchain;

                framebuffers = window_size_dependent_setup(
                    &new_images,
                    render_pass.clone(),
                    &mut dynamic_state,
                );

                recreate_swapchain = false;
            }

            wave_pos[0] += (mouse_pos[0] - wave_pos[0]) * delta_time.as_secs_f32() * 3.;
            wave_pos[1] += (mouse_pos[1] - wave_pos[1]) * delta_time.as_secs_f32() * 3.;

            let (image_num, suboptimal, acquire_future) =
                match swapchain::acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("Failed to acquire next image: {:?}", e),
                };

            if suboptimal {
                recreate_swapchain = true;
            }

            let clear_values = vec![[0.0, 0.0, 0.0, 1.0].into()];

            let mut builder =
                AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family())
                    .unwrap();

            let compute_uniforms = cs::ty::PushConstantData {
                delta_time: delta_time.as_secs_f32(),
                init_image: init_image as _,
                touch_coords: wave_pos,
                touch_force: if mouse_pressed { 10000. } else { 0. },
            };

            init_image = false;

            let render_uniforms = fs::ty::PushConstantData {
                color: [1.0, 1.0, 1.0],
            };

            builder
                .dispatch(
                    [128, 128, 1],
                    compute_pipeline.clone(),
                    compute_set.clone(),
                    compute_uniforms,
                    vec![],
                )
                .unwrap()
                .begin_render_pass(
                    framebuffers[image_num].clone(),
                    SubpassContents::Inline,
                    clear_values,
                )
                .unwrap()
                .draw(
                    pipeline.clone(),
                    &dynamic_state,
                    vertex_buffer.clone(),
                    render_set.clone(),
                    render_uniforms,
                    vec![],
                )
                .unwrap()
                .end_render_pass()
                .unwrap();

            let command_buffer = builder.build().unwrap();

            let future = previous_frame_end
                .take()
                .unwrap()
                .join(acquire_future)
                .then_execute(queue.clone(), command_buffer)
                .unwrap()
                .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                .then_signal_fence_and_flush();

            match future {
                Ok(future) => {
                    previous_frame_end = Some(future.boxed());
                }
                Err(FlushError::OutOfDate) => {
                    recreate_swapchain = true;
                    previous_frame_end = Some(sync::now(device.clone()).boxed());
                }
                Err(e) => {
                    println!("Failed to flush future: {:?}", e);
                    previous_frame_end = Some(sync::now(device.clone()).boxed());
                }
            }

            println!("fps: {:?}", 1. / delta_time.as_secs_f32());
        }
        _ => (),
    });
}

fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState,
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions.width() as f32, dimensions.height() as f32],
        depth_range: 0.0..1.0,
    };
    dynamic_state.viewports = Some(vec![viewport]);

    images
        .iter()
        .map(|image| {
            Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(image.clone())
                    .unwrap()
                    .build()
                    .unwrap(),
            ) as Arc<dyn FramebufferAbstract + Send + Sync>
        })
        .collect::<Vec<_>>()
}

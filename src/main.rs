extern crate sdl2;
extern crate vulkano;
extern crate vulkano_shaders;
extern crate nalgebra as na;
extern crate num_traits;
extern crate imgui;

mod shaders;
mod sendable;
mod renderer;

use vulkano::swapchain;
use vulkano::swapchain::{Swapchain, SurfaceTransform, PresentMode, FullscreenExclusive, ColorSpace, AcquireError, SwapchainCreationError};
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;
use vulkano::buffer::{CpuAccessibleBuffer, BufferUsage, CpuBufferPool};
use std::sync::Arc;
use vulkano::image::{ImageUsage, SwapchainImage};
use vulkano::sync;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::framebuffer::{Subpass, RenderPassAbstract, FramebufferAbstract, Framebuffer};
use vulkano::command_buffer::{DynamicState, AutoCommandBufferBuilder};
use sdl2::video::{WindowContext};
use vulkano::pipeline::viewport::Viewport;
use vulkano::sync::{GpuFuture, FlushError};
use std::rc::Rc;
use crate::sendable::Sendable;
use std::time::Instant;
use vulkano::descriptor::PipelineLayoutAbstract;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use na::{Matrix4, Perspective3, Isometry3, Point3, Vector3};
use std::f32::consts::PI;
use imgui::{Context, Window, Condition, im_str, FontSource, FontConfig, TextureId, DrawCmd, DrawCmdParams};
use crate::renderer::Renderer;


#[derive(Default, Copy, Clone)]
struct Vertex {
    position: [f32; 4],
    color: [f32; 4],
}

impl Vertex {
    fn new(data: [f32; 8]) -> Vertex {
        Vertex {
            position: [data[0], data[1], data[2], data[3]],
            color: [data[4], data[5], data[6], data[7]],
        }
    }
}

vulkano::impl_vertex!(Vertex, position, color);

fn main() {
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    let window = video_subsystem.window("Window", 800, 600)
        .resizable()
        .vulkan()
        .build()
        .unwrap();

    let renderer = Renderer::init_vulkan(&window).expect("Could not init vulkan");
    let (device, queue, surface) = (renderer.device.clone(), renderer.queue.clone(), renderer.surface.clone());

    println!(
        "Using device: {} (type{:?})",
        device.physical_device().name(),
        device.physical_device().ty()
    );

    let mut imgui = Context::create();
    imgui.set_ini_filename(None);

    imgui.io_mut().display_size = [window.size().0 as f32, window.size().1 as f32];
    imgui.fonts().add_font(&[
        FontSource::DefaultFontData {
            config: Some(FontConfig {
                size_pixels: 13.0,
                ..FontConfig::default()
            }),
        },
    ]);

    imgui.io_mut().font_global_scale = 1.0;
    let atlas_texture = imgui.fonts().build_rgba32_texture();

    let (mut swapchain, images) = {
        let caps = surface.capabilities(device.physical_device()).unwrap();

        let alpha = caps.supported_composite_alpha.iter().next().unwrap();

        let format = caps.supported_formats[0].0;

        let dimensions: [u32; 2] = [window.size().0, window.size().1];

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
        ).unwrap()
    };

    let vertex_data: Vec<Vertex> = Vec::from([
        Vertex::new([0.25,  0.25, 0.75, 1.0, 0.0, 0.0, 1.0, 1.0]),
        Vertex::new([0.25, -0.25, 0.75, 1.0, 0.0, 0.0, 1.0, 1.0,]),
        Vertex::new([-0.25,  0.25, 0.75, 1.0, 0.0, 0.0, 1.0, 1.0,]),

        Vertex::new([0.25, -0.25, 0.75, 1.0, 0.0, 0.0, 1.0, 1.0]),
        Vertex::new([-0.25, -0.25, 0.75, 1.0, 0.0, 0.0, 1.0, 1.0,]),
        Vertex::new([-0.25,  0.25, 0.75, 1.0, 0.0, 0.0, 1.0, 1.0,]),

        Vertex::new([0.25,  0.25, -0.75, 1.0, 0.8, 0.8, 0.8, 1.0,]),
        Vertex::new([-0.25,  0.25, -0.75, 1.0, 0.8, 0.8, 0.8, 1.0,]),
        Vertex::new([0.25, -0.25, -0.75, 1.0, 0.8, 0.8, 0.8, 1.0,]),

        Vertex::new([0.25, -0.25, -0.75, 1.0, 0.8, 0.8, 0.8, 1.0,]),
        Vertex::new([-0.25,  0.25, -0.75, 1.0, 0.8, 0.8, 0.8, 1.0,]),
        Vertex::new([-0.25, -0.25, -0.75, 1.0, 0.8, 0.8, 0.8, 1.0]),

        Vertex::new([-0.25,  0.25,  0.75, 1.0, 0.0, 1.0, 0.0, 1.0,]),
        Vertex::new([-0.25, -0.25,  0.75, 1.0, 0.0, 1.0, 0.0, 1.0,]),
        Vertex::new([-0.25, -0.25, -0.75, 1.0, 0.0, 1.0, 0.0, 1.0]),

        Vertex::new([-0.25,  0.25,  0.75, 1.0, 0.0, 1.0, 0.0, 1.0,]),
        Vertex::new([-0.25, -0.25, -0.75, 1.0, 0.0, 1.0, 0.0, 1.0,]),
        Vertex::new([-0.25,  0.25, -0.75, 1.0, 0.0, 1.0, 0.0, 1.0,]),

        Vertex::new([0.25,  0.25,  0.75, 1.0, 0.5, 0.5, 0.0, 1.0,]),
        Vertex::new([0.25, -0.25, -0.75, 1.0, 0.5, 0.5, 0.0, 1.0]),
        Vertex::new([0.25, -0.25,  0.75, 1.0, 0.5, 0.5, 0.0, 1.0,]),

        Vertex::new([0.25,  0.25,  0.75, 1.0, 0.5, 0.5, 0.0, 1.0,]),
        Vertex::new([0.25,  0.25, -0.75, 1.0, 0.5, 0.5, 0.0, 1.0,]),
        Vertex::new([0.25, -0.25, -0.75, 1.0, 0.5, 0.5, 0.0, 1.0,]),

        Vertex::new([0.25,  0.25, -0.75, 1.0, 1.0, 0.0, 0.0, 1.0,]),
        Vertex::new([0.25,  0.25,  0.75, 1.0, 1.0, 0.0, 0.0, 1.0,]),
        Vertex::new([-0.25,  0.25,  0.75, 1.0, 1.0, 0.0, 0.0, 1.0,]),

        Vertex::new([0.25,  0.25, -0.75, 1.0, 1.0, 0.0, 0.0, 1.0,]),
        Vertex::new([-0.25,  0.25,  0.75, 1.0, 1.0, 0.0, 0.0, 1.0,]),
        Vertex::new([-0.25,  0.25, -0.75, 1.0, 1.0, 0.0, 0.0, 1.0,]),

        Vertex::new([0.25, -0.25, -0.75, 1.0, 0.0, 1.0, 1.0, 1.0,]),
        Vertex::new([-0.25, -0.25,  0.75, 1.0, 0.0, 1.0, 1.0, 1.0,]),
        Vertex::new([0.25, -0.25,  0.75, 1.0, 0.0, 1.0, 1.0, 1.0,]),

        Vertex::new([0.25, -0.25, -0.75, 1.0, 0.0, 1.0, 1.0, 1.0,]),
        Vertex::new([-0.25, -0.25, -0.75, 1.0, 0.0, 1.0, 1.0, 1.0,]),
        Vertex::new([-0.25, -0.25,  0.75, 1.0, 0.0, 1.0, 1.0, 1.0])
    ]);

    let vs = shaders::vs::Shader::load(device.clone()).unwrap();
    let fs = shaders::fs::Shader::load(device.clone()).unwrap();

    let render_pass = Arc::new(
        vulkano::single_pass_renderpass!(
            device.clone(),
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
        ).unwrap(),
    );

    let pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_input_single_buffer()
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .cull_mode_back()
            .front_face_clockwise()
            .build(device.clone())
            .unwrap()
    );

    let mut dynamic_state = DynamicState {
        line_width: None,
        viewports: None,
        scissors: None,
        compare_mask: None,
        write_mask: None,
        reference: None
    };

    let mut framebuffers = window_size_dependent_setup(&images[..], render_pass.clone(), &mut dynamic_state);

    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    let mut event_pump = sdl_context.event_pump().unwrap();

    let vertex_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false,
                                                       vertex_data.iter().cloned()).unwrap();

    let uniform_buffer = CpuBufferPool::<shaders::vs::ty::Data>::new(device.clone(), BufferUsage::all());

    let start_time = Instant::now();

    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit {..} | Event::KeyDown { keycode: Some(Keycode::Escape), .. } => {
                    break 'running
                },
                Event::Window { win_event: WindowEvent::Resized(_, _), .. } => {
                    recreate_swapchain = true;
                },
                Event::MouseButtonDown { x, y, .. } => {
                    println!("Click at {:?}, {:?}", x, y);
                }
                _ => {}
            }
        }

        previous_frame_end.as_mut().unwrap().cleanup_finished();

        if recreate_swapchain {
            let dimensions: [u32; 2] = [window.size().0, window.size().1];
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
                &mut dynamic_state
            );

            recreate_swapchain = false;
        }

        let (image_num, suboptimal, acquire_future) =
            match swapchain::acquire_next_image(swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    recreate_swapchain = true;
                    continue;
                }
                Err(e) => panic!("Failed to acquire next image: {:?}", e),
            };

        if suboptimal {
            recreate_swapchain = true;
        }

        let clear_values = vec![[0.0, 0.0, 0.0, 1.0].into()];

        let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(
            device.clone(),
            queue.family(),
        ).unwrap();

        let uniform_buffer_subbuffer = {
            let loop_duration: f32 = 5.0;
            let scale: f32 = std::f64::consts::PI as f32 * 2.0 / loop_duration;

            let elapsed_time: f32 = start_time.elapsed().as_secs_f32();
            let current_time_loop = elapsed_time % loop_duration;

            let x_offset = (current_time_loop * scale).cos() * 0.5;
            let y_offset = (current_time_loop * scale).sin() * 0.5;

            let eye = Point3::new(0.0, 0.0, 2.0);
            let target = Point3::new(0.0, 0.0, 0.0);

            let model = Isometry3::new(na::zero(), na::zero());
            let projection = Perspective3::new(16.0/9.0, PI / 2.0, 0.5, 10.0);
            let view = Isometry3::look_at_rh(&eye, &target, &Vector3::y());

            let model_view = view * model;
            let mat_model_view = model_view.to_homogeneous();

            let vulkan_inverted: Matrix4<f32> = Matrix4::new(
                1.0, 0.0, 0.0, 0.0,
                0.0, -1.0, 0.0, 0.0,
                0.0, 0.0, 0.5, 0.0,
                0.0, 0.0, 0.5, 1.0,
            );

            let uniform_data = shaders::vs::ty::Data {
                offset: [x_offset, y_offset],
                perspectiveMatrix: (vulkan_inverted * projection.as_matrix() * mat_model_view).into(),
            };

            uniform_buffer.next(uniform_data).unwrap()
        };

        let layout = pipeline.descriptor_set_layout(0).unwrap();
        let set = Arc::new(
            PersistentDescriptorSet::start(layout.clone())
                .add_buffer(uniform_buffer_subbuffer)
                .unwrap()
                .build()
                .unwrap(),
        );

        let ui = imgui.frame();
        Window::new(im_str!("Hello World"))
            .size([300.0, 100.0], Condition::FirstUseEver)
            .build(&ui, || {
                ui.text(im_str!("Hello world.1"));
                ui.text(im_str!("こんにちは世界！"));
                ui.text(im_str!("This...is...imgui-rs!"));
                ui.separator();
                let mouse_pos = ui.io().mouse_pos;
                ui.text(format!(
                    "Mouse Position: ({:.1},{:.1})",
                    mouse_pos[0], mouse_pos[1]
                ));
            });
        let draw_data = ui.render();

        // let vertex_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false,
                                                           // vertex_data.iter().cloned()).unwrap();

        let mut index_offset = 0;
        let mut vertex_offset = 0;
        let mut current_texture_id: Option<TextureId> = None;
        let clip_offset = draw_data.display_pos;
        let clip_scale = draw_data.framebuffer_scale;

        let render_pass = builder
            .begin_render_pass(framebuffers[image_num].clone(), false, clear_values)
            .unwrap()
            .draw(
                pipeline.clone(),
                &dynamic_state,
                vertex_buffer.clone(),
                set.clone(),
                (),
            )
            .unwrap();

        let uniform_buffer_subbuffer = {
            let vulkan_inverted: Matrix4<f32> = Matrix4::new(
                1.0, 0.0, 0.0, 0.0,
                0.0, -1.0, 0.0, 0.0,
                0.0, 0.0, 0.5, 0.0,
                0.0, 0.0, 0.5, 1.0,
            );

            let uniform_data = shaders::vs::ty::Data {
                offset: [0.0, 0.0],
                perspectiveMatrix: (vulkan_inverted).into(),
            };

            uniform_buffer.next(uniform_data).unwrap()
        };

        let imgui_set = Arc::new(
            PersistentDescriptorSet::start(layout.clone())
                .add_buffer(uniform_buffer_subbuffer)
                .unwrap()
                .build()
                .unwrap(),
        );

        for draw_list in draw_data.draw_lists() {
            for command in draw_list.commands() {
                match command {
                    DrawCmd::Elements {
                        count,
                        cmd_params:
                        DrawCmdParams {
                            clip_rect,
                            texture_id,
                            vtx_offset,
                            idx_offset,
                        },
                    } => {
                        // scissors
                        if Some(texture_id) != current_texture_id {
                            // texture
                            current_texture_id = Some(texture_id);
                        }

                        // draw indexed
                        let vtx_data = draw_list.vtx_buffer().iter().map(|v| {
                            Vertex {
                                position: [v.pos[0], v.pos[1], 0.0, 1.0],
                                color: [1.0, 0.0, 0.0, 1.0]
                            }
                        });
                        let imgui_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, vtx_data).unwrap();
                        let imgui_idx_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, draw_list.idx_buffer().iter().cloned()).unwrap();

                        render_pass
                            .draw_indexed(
                                pipeline.clone(),
                                &dynamic_state,
                                imgui_buffer.clone(),
                                imgui_idx_buffer.clone(),
                                imgui_set.clone(),
                                ()
                            )
                            .unwrap();
                    },
                    _ => {}
                }
            }
        }

        render_pass
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

        ::std::thread::sleep(::std::time::Duration::new(0, 1_000_000_000u32 / 60));
    }
}

fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Sendable<Rc<WindowContext>>>>],
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState,
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0..1.0,
    };
    dynamic_state.viewports = Some(vec![viewport]);

    images.iter().map(|image| {
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

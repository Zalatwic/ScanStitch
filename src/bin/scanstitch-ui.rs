use clap::Parser;
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use minifb::{Key, Window, WindowOptions};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph};
use ratatui::{Frame, Terminal};
use scanstitch::cli::Cli;
use scanstitch::interactive::{
    self, InteractiveRenderCache, InteractiveRenderControls, PreviewCommand, PreviewFrame,
    PreviewQuality,
};
use scanstitch::pipeline;
use std::io;
use std::thread;
use std::time::{Duration, Instant};

const PREVIEW_WIDTH: usize = 960;
const PREVIEW_HEIGHT: usize = 640;
const HIGH_QUALITY_IDLE_MS: u64 = 350;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SelectedControl {
    Exposure,
    Midpoint,
    Slope,
    ToeLift,
    ShoulderMax,
}

impl SelectedControl {
    fn all() -> [Self; 5] {
        [
            Self::Exposure,
            Self::Midpoint,
            Self::Slope,
            Self::ToeLift,
            Self::ShoulderMax,
        ]
    }

    fn label(self) -> &'static str {
        match self {
            Self::Exposure => "Exposure EV",
            Self::Midpoint => "Midpoint",
            Self::Slope => "Slope",
            Self::ToeLift => "Toe lift",
            Self::ShoulderMax => "Shoulder max",
        }
    }

    fn step(self) -> f64 {
        match self {
            Self::Exposure => 0.10,
            Self::Midpoint => 0.01,
            Self::Slope => 0.10,
            Self::ToeLift => 0.002,
            Self::ShoulderMax => 0.002,
        }
    }
}

struct UiState {
    status: String,
    selected: SelectedControl,
    controls: Option<InteractiveRenderControls>,
    saved_message: Option<String>,
    error_message: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let cli = Cli::parse();
    cli.validate()
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidInput, err))?;

    let (preview_tx, preview_rx) = unbounded::<PreviewCommand>();
    let preview_handle = spawn_preview_window(preview_rx);
    let _ = preview_tx.send(PreviewCommand::Status("processing".to_string()));

    let (cache_tx, cache_rx) = bounded::<Result<InteractiveRenderCache, String>>(1);
    let worker_cli = cli.clone();
    thread::spawn(move || {
        let result =
            pipeline::build_interactive_render_cache(&worker_cli).map_err(|err| err.to_string());
        let _ = cache_tx.send(result);
    });

    let terminal_result = run_terminal_ui(cache_rx, preview_tx.clone());
    let _ = preview_tx.send(PreviewCommand::Exit);
    let _ = preview_handle.join();
    terminal_result
}

fn run_terminal_ui(
    cache_rx: Receiver<Result<InteractiveRenderCache, String>>,
    preview_tx: Sender<PreviewCommand>,
) -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = run_ui_loop(&mut terminal, cache_rx, preview_tx);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    result
}

fn run_ui_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    cache_rx: Receiver<Result<InteractiveRenderCache, String>>,
    preview_tx: Sender<PreviewCommand>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut state = UiState {
        status: "Processing source scans...".to_string(),
        selected: SelectedControl::Exposure,
        controls: None,
        saved_message: None,
        error_message: None,
    };
    let mut cache: Option<InteractiveRenderCache> = None;
    let mut pending_high_quality = false;
    let mut last_edit = Instant::now();

    loop {
        if cache.is_none() {
            match cache_rx.try_recv() {
                Ok(Ok(ready_cache)) => {
                    let mut controls = ready_cache.default_controls();
                    if let Some(path) = ready_cache.cli.review_sidecar.as_deref() {
                        match interactive::load_review_sidecar(path) {
                            Ok(sidecar) => {
                                controls = interactive::controls_with_review_sidecar(
                                    controls,
                                    &ready_cache.auto_tone_params,
                                    &sidecar.sidecar,
                                );
                                state.status = "Ready (review sidecar applied)".to_string();
                            }
                            Err(err) => {
                                state.status = "Ready".to_string();
                                state.error_message =
                                    Some(format!("Review sidecar was not applied: {err}"));
                            }
                        }
                    } else {
                        state.status = "Ready".to_string();
                    }
                    state.controls = Some(controls);
                    send_preview(
                        &preview_tx,
                        &ready_cache,
                        &controls,
                        PreviewQuality::Fast,
                        "ready",
                    );
                    cache = Some(ready_cache);
                    pending_high_quality = true;
                    last_edit = Instant::now();
                }
                Ok(Err(err)) => {
                    state.status = "Processing failed".to_string();
                    state.error_message = Some(err.clone());
                    let _ = preview_tx.send(PreviewCommand::Status("failed".to_string()));
                }
                Err(crossbeam_channel::TryRecvError::Empty) => {}
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    state.status = "Processing failed".to_string();
                    state.error_message = Some("pipeline worker disconnected".to_string());
                    let _ = preview_tx.send(PreviewCommand::Status("failed".to_string()));
                }
            }
        }

        if pending_high_quality
            && last_edit.elapsed() >= Duration::from_millis(HIGH_QUALITY_IDLE_MS)
        {
            if let (Some(cache), Some(controls)) = (&cache, state.controls) {
                send_preview(
                    &preview_tx,
                    cache,
                    &controls,
                    PreviewQuality::High,
                    "preview",
                );
                pending_high_quality = false;
            }
        }

        terminal.draw(|frame| draw_ui(frame, &state))?;

        if !event::poll(Duration::from_millis(50))? {
            continue;
        }

        let Event::Key(key) = event::read()? else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }

        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => break,
            KeyCode::Tab => state.selected = next_control(state.selected),
            KeyCode::BackTab => state.selected = previous_control(state.selected),
            KeyCode::Char('r') => {
                if let Some(cache) = &cache {
                    let controls = cache.default_controls();
                    state.controls = Some(controls);
                    state.status = "Reset to auto tone fit".to_string();
                    state.saved_message = None;
                    state.error_message = None;
                    send_preview(&preview_tx, cache, &controls, PreviewQuality::Fast, "reset");
                    pending_high_quality = true;
                    last_edit = Instant::now();
                }
            }
            KeyCode::Char('s') => {
                if let (Some(cache), Some(controls)) = (&cache, state.controls) {
                    state.status = "Saving full-resolution output...".to_string();
                    terminal.draw(|frame| draw_ui(frame, &state))?;
                    let _ = preview_tx.send(PreviewCommand::Status("saving".to_string()));
                    match pipeline::save_interactive_render(cache, &controls) {
                        Ok(_) => {
                            state.status = "Saved".to_string();
                            let sidecar_suffix =
                                if let Some(path) = cache.cli.write_review_sidecar.as_deref() {
                                    format!(", {}", path.display())
                                } else {
                                    String::new()
                                };
                            state.saved_message = Some(format!(
                                "Wrote {}, report.json{}",
                                cache.output_path.display(),
                                sidecar_suffix
                            ));
                            state.error_message = None;
                            let _ = preview_tx.send(PreviewCommand::Status("saved".to_string()));
                        }
                        Err(err) => {
                            state.status = "Save failed".to_string();
                            state.error_message = Some(err.to_string());
                            let _ =
                                preview_tx.send(PreviewCommand::Status("save failed".to_string()));
                        }
                    }
                }
            }
            KeyCode::Right | KeyCode::Up | KeyCode::Char('+') | KeyCode::Char('=') => {
                let delta = state.selected.step();
                adjust_selected_control(
                    &mut state,
                    &cache,
                    &preview_tx,
                    delta,
                    &mut pending_high_quality,
                    &mut last_edit,
                );
            }
            KeyCode::Left | KeyCode::Down | KeyCode::Char('-') => {
                let delta = -state.selected.step();
                adjust_selected_control(
                    &mut state,
                    &cache,
                    &preview_tx,
                    delta,
                    &mut pending_high_quality,
                    &mut last_edit,
                );
            }
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => break,
            _ => {}
        }
    }

    Ok(())
}

fn draw_ui(frame: &mut Frame, state: &UiState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),
            Constraint::Min(8),
            Constraint::Length(5),
        ])
        .split(frame.area());

    let mut status_lines = vec![Line::from(vec![
        Span::styled("Status: ", Style::default().fg(Color::Cyan)),
        Span::raw(state.status.as_str()),
    ])];
    if let Some(message) = &state.saved_message {
        status_lines.push(Line::from(message.as_str()));
    }
    if let Some(message) = &state.error_message {
        status_lines.push(Line::from(Span::styled(
            message.as_str(),
            Style::default().fg(Color::Red),
        )));
    }
    frame.render_widget(
        Paragraph::new(status_lines).block(Block::default().borders(Borders::ALL).title("Run")),
        chunks[0],
    );

    let control_items = if let Some(controls) = state.controls {
        SelectedControl::all()
            .into_iter()
            .map(|control| {
                let value = control_value(controls, control);
                let marker = if control == state.selected {
                    "> "
                } else {
                    "  "
                };
                let style = if control == state.selected {
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };
                ListItem::new(Line::from(vec![
                    Span::styled(marker, style),
                    Span::styled(format!("{:<14}", control.label()), style),
                    Span::styled(format!("{:>8.4}", value), style),
                ]))
            })
            .collect::<Vec<_>>()
    } else {
        vec![ListItem::new("Waiting for the render cache...")]
    };
    frame.render_widget(
        List::new(control_items).block(Block::default().borders(Borders::ALL).title("Controls")),
        chunks[1],
    );

    let help = Paragraph::new(vec![
        Line::from("Tab selects a control. Arrow keys or +/- adjust the selected value."),
        Line::from("r resets to auto, s saves output.tiff and report.json, q quits."),
    ])
    .block(Block::default().borders(Borders::ALL).title("Keys"));
    frame.render_widget(help, chunks[2]);
}

fn adjust_selected_control(
    state: &mut UiState,
    cache: &Option<InteractiveRenderCache>,
    preview_tx: &Sender<PreviewCommand>,
    delta: f64,
    pending_high_quality: &mut bool,
    last_edit: &mut Instant,
) {
    let (Some(cache), Some(mut controls)) = (cache, state.controls) else {
        return;
    };

    match state.selected {
        SelectedControl::Exposure => {
            controls.exposure_ev = (controls.exposure_ev + delta).clamp(-4.0, 4.0)
        }
        SelectedControl::Midpoint => {
            controls.midpoint = (controls.midpoint + delta).clamp(0.001, 0.999)
        }
        SelectedControl::Slope => controls.slope = (controls.slope + delta).clamp(0.1, 16.0),
        SelectedControl::ToeLift => {
            controls.toe_lift = (controls.toe_lift + delta).clamp(0.0, 0.25)
        }
        SelectedControl::ShoulderMax => {
            controls.shoulder_max = (controls.shoulder_max + delta).clamp(0.5, 1.0)
        }
    }
    state.controls = Some(controls);
    state.saved_message = None;
    state.error_message = None;
    state.status = "Preview updated".to_string();
    send_preview(
        preview_tx,
        cache,
        &controls,
        PreviewQuality::Fast,
        "preview",
    );
    *pending_high_quality = true;
    *last_edit = Instant::now();
}

fn control_value(controls: InteractiveRenderControls, control: SelectedControl) -> f64 {
    match control {
        SelectedControl::Exposure => controls.exposure_ev,
        SelectedControl::Midpoint => controls.midpoint,
        SelectedControl::Slope => controls.slope,
        SelectedControl::ToeLift => controls.toe_lift,
        SelectedControl::ShoulderMax => controls.shoulder_max,
    }
}

fn next_control(selected: SelectedControl) -> SelectedControl {
    let controls = SelectedControl::all();
    let index = controls
        .iter()
        .position(|control| *control == selected)
        .unwrap_or(0);
    controls[(index + 1) % controls.len()]
}

fn previous_control(selected: SelectedControl) -> SelectedControl {
    let controls = SelectedControl::all();
    let index = controls
        .iter()
        .position(|control| *control == selected)
        .unwrap_or(0);
    controls[(index + controls.len() - 1) % controls.len()]
}

fn send_preview(
    preview_tx: &Sender<PreviewCommand>,
    cache: &InteractiveRenderCache,
    controls: &InteractiveRenderControls,
    quality: PreviewQuality,
    status: &'static str,
) {
    let frame = interactive::render_preview_frame(
        cache,
        controls,
        PREVIEW_WIDTH,
        PREVIEW_HEIGHT,
        quality,
        status,
    );
    let _ = preview_tx.send(PreviewCommand::Frame(frame));
}

fn spawn_preview_window(rx: Receiver<PreviewCommand>) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut window = match Window::new(
            "scanstitch preview - processing",
            PREVIEW_WIDTH,
            PREVIEW_HEIGHT,
            WindowOptions {
                resize: true,
                ..WindowOptions::default()
            },
        ) {
            Ok(window) => window,
            Err(err) => {
                eprintln!("failed to open preview window: {err}");
                return;
            }
        };
        window.set_target_fps(30);

        let mut canvas = processing_canvas();
        while window.is_open() && !window.is_key_down(Key::Escape) {
            while let Ok(command) = rx.try_recv() {
                match command {
                    PreviewCommand::Status(status) => {
                        window.set_title(&format!("scanstitch preview - {status}"));
                    }
                    PreviewCommand::Frame(frame) => {
                        window.set_title(&format!("scanstitch preview - {}", frame.status));
                        canvas = frame_to_canvas(&frame);
                    }
                    PreviewCommand::Exit => return,
                }
            }

            if window
                .update_with_buffer(&canvas, PREVIEW_WIDTH, PREVIEW_HEIGHT)
                .is_err()
            {
                return;
            }
        }
    })
}

fn processing_canvas() -> Vec<u32> {
    let mut pixels = vec![0x181818; PREVIEW_WIDTH * PREVIEW_HEIGHT];
    let band_top = PREVIEW_HEIGHT / 2 - 12;
    let band_bottom = PREVIEW_HEIGHT / 2 + 12;
    for y in band_top..band_bottom {
        for x in PREVIEW_WIDTH / 5..PREVIEW_WIDTH * 4 / 5 {
            let t = (x - PREVIEW_WIDTH / 5) as f64 / (PREVIEW_WIDTH * 3 / 5).max(1) as f64;
            let value = (70.0 + 90.0 * t).round() as u32;
            pixels[y * PREVIEW_WIDTH + x] = (value << 16) | (value << 8) | value;
        }
    }
    pixels
}

fn frame_to_canvas(frame: &PreviewFrame) -> Vec<u32> {
    let mut canvas = vec![0x101010; PREVIEW_WIDTH * PREVIEW_HEIGHT];
    let copy_width = frame.width.min(PREVIEW_WIDTH);
    let copy_height = frame.height.min(PREVIEW_HEIGHT);
    let left = (PREVIEW_WIDTH - copy_width) / 2;
    let top = (PREVIEW_HEIGHT - copy_height) / 2;

    for y in 0..copy_height {
        let source_row = y * frame.width;
        let target_row = (top + y) * PREVIEW_WIDTH + left;
        canvas[target_row..target_row + copy_width]
            .copy_from_slice(&frame.pixels[source_row..source_row + copy_width]);
    }

    canvas
}

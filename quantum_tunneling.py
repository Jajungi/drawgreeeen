import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use('TkAgg')  # Windows GUI 백엔드
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# 1. 물리 상수 및 공간 격자
ħ = 1.0; m = 1.0
L = 200.0; Nx = 1000
dx = L / Nx
x  = np.linspace(0, L, Nx)

# 2. 초기 웨이브패킷 (가우시안)
def init_wavepacket(x0, k0, sigma):
    norm = 1/(2*np.pi*sigma**2)**0.25
    return norm * np.exp(-(x-x0)**2/(4*sigma**2)) * np.exp(1j*k0*x)

# 3. 구간별 상수 포텐셜
def piecewise_potential(V_left, V_barrier, V_right, a, b):
    V = np.full_like(x, V_left)
    V[(x>=a)&(x<=b)] = V_barrier
    V[x > b]         = V_right
    return V

# 4. 해밀토니안 구성
def build_hamiltonian(V):
    main = ħ**2/(m*dx**2) + V
    off  = -ħ**2/(2*m*dx**2)
    return sp.diags([main, off*np.ones(Nx-1), off*np.ones(Nx-1)],
                    [0, -1, 1], format='csr')

# 5. Crank–Nicolson 전파 연산자
def build_cn_operators(H, dt):
    I = sp.eye(Nx, format='csr')
    A = I + 1j*dt/(2*ħ)*H
    B = I - 1j*dt/(2*ħ)*H
    return spla.factorized(A), B

# 6. 시뮬레이션 파라미터
x0, sigma = 0.2*L, 5.0
E0        = 0.5
dt        = 1.0
t_max     = 2100.0
steps     = int(t_max / dt)
timer_interval = 5      # ms
frame_skip     = 20     # 프레임 스킵

# 7. Figure & Axes: 파동 + 확률밀도
fig = plt.figure(figsize=(10,6))
ax_wave = fig.add_subplot(2,1,1)
ax_den  = fig.add_subplot(2,1,2)
plt.subplots_adjust(left=0.05, right=0.72, bottom=0.05, top=0.95)

# 파동(실수부) + 포텐셜(정규화)
line_wave,      = ax_wave.plot([], [], lw=2, label='Re ψ')
line_barrier_w, = ax_wave.plot([], [], 'k--', lw=1, label='V normalized')
ax_wave.set_xlim(0, L); ax_wave.set_ylim(-1,1)
ax_wave.legend(loc='upper right')
ax_wave.set_ylabel('Re ψ')

# 확률밀도 + 포텐셜
line_den,         = ax_den.plot([], [], lw=2, label='|ψ|²')
line_barrier_den, = ax_den.plot([], [], 'k--', lw=1, label='V normalized')
ax_den.set_xlim(0, L); ax_den.set_ylim(0,0.1)
ax_den.legend(loc='upper right')
ax_den.set_xlabel('x'); ax_den.set_ylabel('|ψ|²')

# 8. 초기 파동
k0_init = np.sqrt(2*m*E0)/ħ
psi      = init_wavepacket(x0, k0_init, sigma)
frame    = 0
playing  = False

# 9. UI 레이아웃: 우측 컨트롤 패널
ctrl_x      = 0.75
ctrl_w      = 0.22
slider_h    = 0.04
slider_sp   = 0.06
y0          = 0.95 - slider_h

ax_Vleft   = fig.add_axes([ctrl_x, y0,       ctrl_w, slider_h])
ax_Vbar    = fig.add_axes([ctrl_x, y0-slider_sp, ctrl_w, slider_h])
ax_Vright  = fig.add_axes([ctrl_x, y0-2*slider_sp, ctrl_w, slider_h])
ax_width   = fig.add_axes([ctrl_x, y0-3*slider_sp, ctrl_w, slider_h])
ax_energy  = fig.add_axes([ctrl_x, y0-4*slider_sp, ctrl_w, slider_h])
ax_play    = fig.add_axes([ctrl_x, y0-5*slider_sp-0.02, ctrl_w/2-0.01, slider_h])
ax_reset   = fig.add_axes([ctrl_x+ctrl_w/2+0.01, y0-5*slider_sp-0.02, ctrl_w/2-0.01, slider_h])

slider_Vleft  = Slider(ax_Vleft,  'V left',    0.0, 1.0, valinit=0.0)
slider_Vbar   = Slider(ax_Vbar,   'V barrier', 0.0, 1.0, valinit=0.0)
slider_Vright = Slider(ax_Vright, 'V right',   0.0, 1.0, valinit=0.0)
slider_width  = Slider(ax_width,  'Barrier W', 1.0, 50.0, valinit=10.0)
slider_energy = Slider(ax_energy, 'Energy',    0.0, 1.0,  valinit=0.5)  # 에너지도 0~1
btn_play      = Button(ax_play,  'Play')
btn_reset     = Button(ax_reset, 'Restart')

# 10. 한 프레임 계산 및 갱신
def draw_frame(i):
    global psi
    a = 0.4*L; b = a + slider_width.val
    V = piecewise_potential(slider_Vleft.val,
                            slider_Vbar.val,
                            slider_Vright.val,
                            a, b)
    if i == 0:
        k0 = np.sqrt(2*m*slider_energy.val)/ħ
        psi[:] = init_wavepacket(x0, k0, sigma)

    H = build_hamiltonian(V)
    A, B = build_cn_operators(H, dt)
    psi[:] = A(B.dot(psi))

    real_part = np.real(psi)
    density   = np.abs(psi)**2
    barrier   = V / (np.max(V)+1e-8)

    line_wave.set_data(x, real_part)
    line_barrier_w.set_data(x, barrier)
    line_den.set_data(x, density)
    line_barrier_den.set_data(x, barrier)

# 11. Timer 콜백 (반복)
timer = fig.canvas.new_timer(interval=timer_interval)
def on_timer():
    global frame
    draw_frame(frame)
    frame += frame_skip
    if frame >= steps:
        frame = 0
    fig.canvas.draw_idle()
timer.add_callback(on_timer)

# 12. 버튼 콜백
def toggle_play(event):
    global playing
    if not playing:
        timer.start(); btn_play.label.set_text('Pause')
    else:
        timer.stop(); btn_play.label.set_text('Play')
    playing = not playing

def do_reset(event):
    global frame, playing
    timer.stop()
    frame = 0; playing = False
    btn_play.label.set_text('Play')
    draw_frame(0); fig.canvas.draw_idle()

btn_play.on_clicked(toggle_play)
btn_reset.on_clicked(do_reset)

# 13. 슬라이더 조작 시 자동 일시정지
def pause_on_change(val):
    global playing
    if playing:
        timer.stop(); btn_play.label.set_text('Play'); playing = False

for s in (slider_Vleft, slider_Vbar, slider_Vright,
          slider_width, slider_energy):
    s.on_changed(pause_on_change)

# 14. 초기 프레임 & 실행
draw_frame(0)
fig.canvas.draw_idle()
plt.show()

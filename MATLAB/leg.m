%% 仿蜘蛛六足机器人单腿工作空间可视化
% 作者：基于用户提供的参数
% 日期：当前日期

clear; clc; close all;

%% 1. 定义DH参数（单位：cm）
% 根据用户提供的参数：
% joint1与joint2之间连杆距离: 23.5cm × 2 = 47cm
% joint2到joint3连杆: 65.5cm
% 末端连杆: 105cm

% DH参数表: [α, a, d, θ, σ]
% α: 连杆扭转角 (rad)
% a: 连杆长度 (cm)
% d: 连杆偏移 (cm)
% θ: 关节角度 (rad)
% σ: 关节类型 (0为旋转，1为移动)

L(1) = Link('alpha', -pi/2, 'a', 47, 'd', 0, 'offset', 0, 'standard');
L(2) = Link('alpha', 0, 'a', 65, 'd', 0, 'offset', 0, 'standard');
L(3) = Link('alpha', 0, 'a', 105, 'd', 0, 'offset', 0, 'standard');

%% 2. 创建机器人对象
robot = SerialLink(L, 'name', 'Spider Leg');

% 显示机器人参数
disp('=== 机器人参数 ===');
disp(robot);
fprintf('\nDH参数表:\n');
robot.display();

%% 3. 设置关节限位
% 假设关节运动范围（根据典型蜘蛛腿运动范围设置）
qlim1 = [-pi/2, pi/2];     % 关节1: ±90度侧摆
qlim2 = [pi/2, 0];        % 关节2: -90到0度（大腿）
qlim3 = [-pi, 0];          % 关节3: -180到0度（小腿）

robot.links(1).qlim = qlim1;
robot.links(2).qlim = qlim2;
robot.links(3).qlim = qlim3;

%% 4. 可视化机器人结构
figure('Name', '机器人结构', 'Position', [100, 100, 1200, 500]);

% 子图1: 默认位置
subplot(1,2,1);
robot.plot([0, 0, 0], 'workspace', [-200 200 -200 200 -200 50], ...
           'scale', 0.5, 'view', [30, 20]);
title('默认位置 (θ=[0,0,0])');
xlabel('X (cm)'); ylabel('Y (cm)'); zlabel('Z (cm)');
grid on;

% 子图2: 典型行走位置
subplot(1,2,2);
q_walk = [pi/6, -pi/4, -pi/2];  % 示例行走姿态
robot.plot(q_walk, 'workspace', [-200 200 -200 200 -200 50], ...
           'scale', 0.5, 'view', [30, 20]);
title('典型行走姿态');
xlabel('X (cm)'); ylabel('Y (cm)'); zlabel('Z (cm)');
grid on;

%% 5. 工作空间分析
fprintf('\n=== 工作空间分析 ===\n');

% 计算可达工作空间
n_samples = 10000;  % 采样点数
workspace_points = zeros(n_samples, 3);

% 生成随机关节角度
q1_random = qlim1(1) + (qlim1(2)-qlim1(1)) * rand(n_samples, 1);
q2_random = qlim2(1) + (qlim2(2)-qlim2(1)) * rand(n_samples, 1);
q3_random = qlim3(1) + (qlim3(2)-qlim3(1)) * rand(n_samples, 1);

% 计算末端位置
for i = 1:n_samples
    T = robot.fkine([q1_random(i), q2_random(i), q3_random(i)]);
    workspace_points(i, :) = T.t';
end

%% 6. 工作空间三维可视化
figure('Name', '工作空间三维可视化', 'Position', [100, 100, 1400, 600]);

% 子图1: 三维散点图
subplot(1,2,1);
scatter3(workspace_points(:,1), workspace_points(:,2), workspace_points(:,3), ...
         5, workspace_points(:,3), 'filled');
colormap(jet);
colorbar;
xlabel('X (cm)'); ylabel('Y (cm)'); zlabel('Z (cm)');
title('工作空间三维分布');
grid on;
axis equal;
view(3);

% 子图2: 投影视图
subplot(2,2,2);
scatter(workspace_points(:,1), workspace_points(:,2), 5, 'b', 'filled');
xlabel('X (cm)'); ylabel('Y (cm)');
title('XY平面投影');
grid on;
axis equal;

subplot(2,2,4);
scatter(workspace_points(:,1), workspace_points(:,3), 5, 'r', 'filled');
xlabel('X (cm)'); ylabel('Z (cm)');
title('XZ平面投影');
grid on;
axis equal;

%% 7. 工作空间边界分析
figure('Name', '工作空间边界分析', 'Position', [100, 100, 1200, 400]);

% 计算工作空间极值
x_min = min(workspace_points(:,1));
x_max = max(workspace_points(:,1));
y_min = min(workspace_points(:,2));
y_max = max(workspace_points(:,2));
z_min = min(workspace_points(:,3));
z_max = max(workspace_points(:,3));

fprintf('工作空间范围:\n');
fprintf('X: [%.2f, %.2f] cm\n', x_min, x_max);
fprintf('Y: [%.2f, %.2f] cm\n', y_min, y_max);
fprintf('Z: [%.2f, %.2f] cm\n', z_min, z_max);
fprintf('工作空间体积估算: %.2f cm³\n', ...
        (x_max-x_min)*(y_max-y_min)*(z_max-z_min));

% 绘制工作空间边界
subplot(1,3,1);
histogram(workspace_points(:,1), 50);
xlabel('X坐标 (cm)'); ylabel('点数');
title('X方向分布');

subplot(1,3,2);
histogram(workspace_points(:,2), 50);
xlabel('Y坐标 (cm)'); ylabel('点数');
title('Y方向分布');

subplot(1,3,3);
histogram(workspace_points(:,3), 50);
xlabel('Z坐标 (cm)'); ylabel('点数');
title('Z方向分布');

%% 8. 轨迹规划示例
figure('Name', '轨迹规划示例', 'Position', [100, 100, 1200, 400]);

% 定义轨迹点
t = linspace(0, 1, 100)';

% 示例轨迹1: 抬腿动作
q_traj1 = [zeros(100,1), ...               % 关节1保持
           -pi/4*ones(100,1), ...          % 关节2固定
           linspace(0, -pi/2, 100)'];      % 关节3弯曲

% 示例轨迹2: 摆动动作
q_traj2 = [linspace(-pi/6, pi/6, 100)', ...  % 关节1摆动
           linspace(-pi/6, -pi/3, 100)', ... % 关节2变化
           linspace(-pi/3, -pi/2, 100)'];    % 关节3变化

% 计算末端轨迹
pos_traj1 = zeros(100, 3);
pos_traj2 = zeros(100, 3);

for i = 1:100
    T1 = robot.fkine(q_traj1(i,:));
    T2 = robot.fkine(q_traj2(i,:));
    pos_traj1(i,:) = T1.t';
    pos_traj2(i,:) = T2.t';
end

% 绘制轨迹
subplot(1,2,1);
plot3(pos_traj1(:,1), pos_traj1(:,2), pos_traj1(:,3), 'b-', 'LineWidth', 2);
hold on;
robot.plot(q_traj1(end,:), 'workspace', [-200 200 -200 200 -200 50], ...
           'scale', 0.3, 'view', [30, 20], 'noname');
title('抬腿轨迹');
xlabel('X (cm)'); ylabel('Y (cm)'); zlabel('Z (cm)');
grid on;
axis equal;

subplot(1,2,2);
plot3(pos_traj2(:,1), pos_traj2(:,2), pos_traj2(:,3), 'r-', 'LineWidth', 2);
hold on;
robot.plot(q_traj2(end,:), 'workspace', [-200 200 -200 200 -200 50], ...
           'scale', 0.3, 'view', [30, 20], 'noname');
title('摆动轨迹');
xlabel('X (cm)'); ylabel('Y (cm)'); zlabel('Z (cm)');
grid on;
axis equal;

%% 9. 导出机器人信息
% 生成详细报告
fprintf('\n=== 机器人单腿详细信息 ===\n');
fprintf('总连杆长度: %.2f cm\n', 47 + 65.5 + 105);
fprintf('关节数量: %d\n', robot.n);
fprintf('自由度: %d\n', robot.n);
fprintf('末端执行器可达最大距离: %.2f cm\n', sqrt((47+65.5+105)^2));

% 保存工作空间数据
save('spider_leg_workspace.mat', 'workspace_points', 'robot');

disp('分析完成！');
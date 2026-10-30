function [sh_coeffs, offset, fitted_surface] = SHP_Surface_Fitting(antenna_pos, measured_distances, max_degree)
% SHP_SURFACE_FITTING Fits a 3D surface using Spherical Harmonic Parametric descriptor.
%
% Inputs:
%   antenna_pos        : Nx3 matrix containing [X, Y, Z] coordinates of the sensors
%   measured_distances : Nx1 vector of estimated radial distances from sensors to skin
%   max_degree         : Maximum degree (N) of Spherical Harmonics (e.g., N=3)
%
% Outputs:
%   sh_coeffs          : Vector of computed SH coefficients
%   offset             : Optimized global center offset [x0, y0, z0]
%   fitted_surface     : Struct containing the high-resolution grid visualization data

    N_sensors = size(antenna_pos, 1);
    
    % Step 1: Compute total number of SH coefficients for degree N
    % Total coefficients = (N + 1)^2
    num_sh_vars = (max_degree + 1)^2;
    
    % Step 2: Initialize optimization variables
    % [sh_coefficients (num_sh_vars x 1), offset_x, offset_y, offset_z]
    % Initial guess: Sphere with mean radius, offset centered at geometric mean
    mean_center = mean(antenna_pos, 1);
    estimated_radii = zeros(N_sensors, 1);
    for i = 1:N_sensors
        estimated_radii(i) = norm(antenna_pos(i, :) - mean_center) - measured_distances(i);
    end
    init_radius = mean(estimated_radii);
    
    init_sh = zeros(num_sh_vars, 1);
    init_sh(1) = init_radius * sqrt(4 * pi); % Real SH base term Y_0^0 constant
    
    init_params = [init_sh; mean_center(:)];
    
    % Step 3: Configure optimization parameters
    options = optimoptions('fmincon', ...
        'Display', 'iter-detailed', ...
        'Algorithm', 'interior-point', ...
        'MaxFunctionEvaluations', 30000, ...
        'MaxIterations', 1000, ...
        'OptimalityTolerance', 1e-6, ...
        'StepTolerance', 1e-6);
    
    % Step 4: Execute constrained optimization
    % Objective minimizes the error between model boundary and physics-driven tangency points
    cost_func = @(p) SHP_Objective(p, antenna_pos, measured_distances, max_degree);
    
    % Optional boundary constraints to stabilize high-degree perturbations
    lb = [-Inf(num_sh_vars, 1); mean_center(:) - 30]; 
    ub = [ Inf(num_sh_vars, 1); mean_center(:) + 30];
    
    fprintf('Starting Physics-Driven SHP Boundary Optimization...\n');
    [opt_params, ~] = fmincon(cost_func, init_params, [], [], [], [], lb, ub, [], options);
    
    % Step 5: Extract optimized parameters
    sh_coeffs = opt_params(1:num_sh_vars);
    offset = opt_params(num_sh_vars+1:end)';
    
    % Step 6: Generate high-resolution continuous surface grid for validation
    [theta_grid, phi_grid] = meshgrid(linspace(0, 2*pi, 100), linspace(0, pi, 50));
    Y_matrix = Compute_SH_Basis(theta_grid(:), phi_grid(:), max_degree);
    R_fitted = reshape(Y_matrix * sh_coeffs, size(theta_grid));
    
    % Convert parametric surface back to Cartesian coordinates with spatial translation
    fitted_surface.X = R_fitted .* sin(phi_grid) .* cos(theta_grid) + offset(1);
    fitted_surface.Y = R_fitted .* sin(phi_grid) .* sin(theta_grid) + offset(2);
    fitted_surface.Z = R_fitted .* cos(phi_grid) + offset(3);
    fitted_surface.R = R_fitted;
    fitted_surface.Theta = theta_grid;
    fitted_surface.Phi = phi_grid;
    
    % Plot the reconstructed manifold
    figure;
    surf(fitted_surface.X, fitted_surface.Y, fitted_surface.Z, fitted_surface.R, 'EdgeColor', 'none');
    colormap('jet'); title('Reconstructed 3D Analytical Manifold via SHP');
    xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)'); axis equal; grid on; colorbar;
end

%% --- Local Objective Function ---
function total_loss = SHP_Objective(params, antenna_pos, measured_distances, max_degree)
    num_sh_vars = (max_degree + 1)^2;
    sh_coeffs = params(1:num_sh_vars);
    offset = params(num_sh_vars+1:end)';
    
    % Translate antenna array relative to the estimated moving center O
    translated_antennas = antenna_pos - repmat(offset, size(antenna_pos, 1), 1);
    
    % Transform shifted positions into local spherical coordinates
    [ant_theta, ant_phi, ant_R] = Cartesian_To_Spherical(translated_antennas);
    
    % Compute the theoretical radius of the SH surface at specific sensor trajectories
    Y_basis = Compute_SH_Basis(ant_theta, ant_phi, max_degree);
    R_surface = Y_basis * sh_coeffs;
    
    % Calculate the calculated radial distance to the surface
    % The physical distance from the sensor to the skin boundary
    calculated_distances = ant_R - R_surface;
    
    % Mean squared error of the spatial tangency condition
    residual_error = sum((calculated_distances - measured_distances).^2);
    
    % Geometric regularization: Penalize high-frequency spatial oscillations
    % This acts as a spatial low-pass filter to maintain surface convexity
    reg_weight = 1e-4;
    reg_penalty = 0;
    idx = 1;
    for n = 0:max_degree
        for m = -n:n
            reg_penalty = reg_penalty + (n * (n + 1))^2 * sh_coeffs(idx)^2;
            idx = idx + 1;
        end
    end
    
    total_loss = residual_error + reg_weight * reg_penalty;
end

%% --- Local Spherical Transformation Helper ---
function [theta, phi, R] = Cartesian_To_Spherical(pts)
    R = sqrt(sum(pts.^2, 2));
    phi = acos(pts(:, 3) ./ max(R, 1e-9)); % Polar/Elevation angle [0, pi]
    theta = atan2(pts(:, 2), pts(:, 1));   % Azimuthal angle [-pi, pi]
    theta(theta < 0) = theta(theta < 0) + 2*pi; % Remap to [0, 2pi]
end

%% --- Real Spherical Harmonic Basis Generator ---
function Y = Compute_SH_Basis(theta, phi, max_degree)
    N_points = length(theta);
    num_vars = (max_degree + 1)^2;
    Y = zeros(N_points, num_vars);
    
    col_idx = 1;
    for n = 0:max_degree
        % Compute normalization and Legendre polynomials
        P = legendre(n, cos(phi'));
        
        for m = -n:n
            abs_m = abs(m);
            % Extract correct row from MATLAB's legendre output vector
            P_nm = P(abs_m + 1, :)';
            
            % Compute normalization factor N_n^m
            factor = sqrt(((2*n + 1) * factorial(n - abs_m)) / (4 * pi * factorial(n + abs_m)));
            Y_norm = factor * P_nm;
            
            % Compute real spherical harmonics transformation
            if m > 0
                Y(:, col_idx) = Y_norm .* cos(m * theta) * sqrt(2);
            elseif m < 0
                Y(:, col_idx) = Y_norm .* sin(abs_m * theta) * sqrt(2);
            else
                Y(:, col_idx) = Y_norm;
            end
            col_idx = col_idx + 1;
        end
    end
end
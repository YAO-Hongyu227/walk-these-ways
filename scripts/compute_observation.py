def compute_observations(self):
        """ Computes observations
        """
        is_debugging = False
        if is_debugging:
            print('Current Observation Includes:')

        if self.cfg.env.observe_command:
            self.obs_buf = torch.cat((self.projected_gravity,
                                      self.commands * self.commands_scale,
                                      (self.dof_pos[:, :self.num_actuated_dof] - self.default_dof_pos[:,:self.num_actuated_dof]) * self.obs_scales.dof_pos,
                                      self.dof_vel[:, :self.num_actuated_dof] * self.obs_scales.dof_vel,
                                      self.actions
                                      ), dim=-1)
            if is_debugging:
                print('self.projected_gravity',self.projected_gravity.shape)
                print('self.commands * self.commands_scale',(self.commands * self.commands_scale).shape)
                print('self.commands_scales',self.commands_scale)
                print('one command', self.commands[0,:])
                print('(self.dof_pos[:, :self.num_actuated_dof] - self.default_dof_pos[:,:self.num_actuated_dof]) * self.obs_scales.dof_pos',((self.dof_pos[:, :self.num_actuated_dof] - self.default_dof_pos[:,:self.num_actuated_dof]) * self.obs_scales.dof_pos).shape)
                print('self.dof_vel[:, :self.num_actuated_dof] * self.obs_scales.dof_vel',(self.dof_vel[:, :self.num_actuated_dof] * self.obs_scales.dof_vel).shape)
                print('self.actions',self.actions.shape)
            

        if self.cfg.env.observe_two_prev_actions:
            self.obs_buf = torch.cat((self.obs_buf,
                                      self.last_actions), dim=-1)
            if is_debugging:
                print('self.last_actions',self.last_actions.shape)


        if self.cfg.env.observe_clock_inputs:
            self.obs_buf = torch.cat((self.obs_buf,
                                      self.clock_inputs), dim=-1)
            if is_debugging:    
                print('self.clock_inputs',self.clock_inputs.shape)

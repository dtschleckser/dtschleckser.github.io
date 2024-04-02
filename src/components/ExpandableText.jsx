import React from 'react';
import { Box, Collapse } from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';

const ExpandableText = ({ children, headerText }) => {
  const [opened, { toggle }] = useDisclosure(false);

  return (
    <Box>
      <h3 onClick={toggle} style={{ cursor: 'pointer', userSelect: 'none' }}>
        {opened ? '🡣' : '🡢'} {headerText}
      </h3>
      <Collapse in={opened}>
        {children}
      </Collapse>
    </Box>
  );
};

export default ExpandableText;